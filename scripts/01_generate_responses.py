#!/usr/bin/env python
"""Step 1 - Sample one response per problem with vLLM and grade it.

Inputs
    --model           HF model id (e.g. hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo)
    --benchmark       math500, gsm8k or arc
    --data_dir        directory holding the benchmark files (default ./data)

Output
    <output_dir>/<benchmark>/test_<prompt_type>_<max_samples>_seed<seed>_t<temperature>_s0_e-1.jsonl
    (with the defaults: test_<prompt_type>_-1_seed0_t1.0_s0_e-1.jsonl), plus ..._metrics.json

Each line holds the response (``code``), the extracted answer (``pred``) and its
``math_equal`` score (``score``); see ``reward_subsystem/outputs.py``. Later steps
label a response correct iff ``pred`` equals the reference answer as a string.
The step is skipped if the ``_metrics.json`` file already exists.

Example
    python scripts/01_generate_responses.py --model hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo \
        --benchmark math500 --output_dir <ARTIFACT_DIR>/outputs/output_7B_simplerl_16000tk
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from reward_subsystem.answer_parsing import choice_answer_clean, extract_answer, strip_string
from reward_subsystem.benchmarks import (
    SUPPORTED_BENCHMARKS, construct_prompt, load_benchmark, parse_ground_truth, parse_question,
)
from reward_subsystem.grading import grade_predictions
from reward_subsystem.outputs import output_filename, write_jsonl

# System prompt used when --apply_chat_template is set (non-Qwen models).
GENERATION_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it "
    "step by step. The assistant first thinks about the reasoning process in the mind and then provides "
    "the user with the answer, i.e., reasoning process here ... the answer is \\boxed{answer}\n\n"
)
STOP_WORDS = ["</s>", "<|im_end|>", "<|endoftext|>"]
# Benchmark fields copied into the output records when present.
EXTRA_FIELDS = ["level", "type", "unit", "solution_type", "choices", "solution", "ques_type",
                "ans_type", "answer_type", "dataset", "subfield", "filed", "theorem", "answer"]
CHOICES = ["A", "B", "C", "D", "E"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--benchmark", default="math500", choices=SUPPORTED_BENCHMARKS)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--prompt_type", default="qwen-boxed", choices=["qwen-boxed", "llama-boxed"],
                   help="qwen-boxed for Qwen models; llama-boxed (with --apply_chat_template) for the others.")
    p.add_argument("--apply_chat_template", action="store_true",
                   help="Wrap the prompt in the tokenizer's chat template (used for non-Qwen models).")
    p.add_argument("--max_samples", type=int, default=-1, help="Use only the first N problems (-1 = all).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=-1)
    p.add_argument("--presence_penalty", type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=16000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tensor_parallel_size", type=int, default=None,
                   help="Default: all visible GPUs (must divide the number of attention heads).")
    return p.parse_args()


def predict(response: str):
    """Extracted, normalised answer of a response (``None`` for an empty response)."""
    if not response or response == "error":
        return None
    return strip_string(extract_answer(response))


def clean_multiple_choice(pred, gt: str, response: str):
    """Map free-form predictions to option letters when the reference is an option letter."""
    def is_multi_choice(answer):
        return all(c in CHOICES for c in answer)

    if gt in CHOICES and pred not in CHOICES:
        return choice_answer_clean(response)
    if is_multi_choice(gt) and not is_multi_choice(pred):
        return "".join(c for c in pred if c in CHOICES)
    return pred


def accuracy_summary(records, n_timeouts: int) -> dict:
    """``math_equal`` accuracy in percent (rounded to 0.1), overall and per ``type`` if present."""
    summary = {
        "num_samples": len(records),
        "num_scores": len(records),
        "timeout_samples": n_timeouts,
        "empty_samples": len([r for r in records if not r["pred"][0]]),
        "acc": float(np.round(np.array([r["score"][0] for r in records]).mean() * 100, decimals=1)),
    }
    if "type" in records[0]:
        type_scores = {}
        for r in records:
            type_scores.setdefault(r["type"], []).append(r["score"][0])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        summary["type_acc"] = dict(sorted(type_scores.items()))
    return summary


def generate(llm, tokenizer, args):
    examples = load_benchmark(args.benchmark, args.data_dir)
    if args.max_samples > 0:
        examples = examples[:args.max_samples]

    # 1. Prompts and reference answers.
    records, prompts = [], []
    for example in examples:
        example["question"] = parse_question(example, args.benchmark)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, args.benchmark)
        record = {"idx": example["idx"], "question": example["question"], "gt_cot": gt_cot, "gt": gt_ans}
        record.update({k: example[k] for k in EXTRA_FIELDS if k in example})
        records.append(record)
        prompts.append(construct_prompt(example["question"], args.prompt_type))

    if args.apply_chat_template:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                 {"role": "user", "content": prompt.strip()}],
                tokenize=False, add_generation_prompt=True)
            for prompt in prompts
        ]

    # 2. Sampling (a single vLLM call; outputs re-ordered by request id).
    from vllm import SamplingParams  # imported lazily so that --help works without a GPU

    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        presence_penalty=args.presence_penalty, max_tokens=args.max_new_tokens, n=1, stop=STOP_WORDS,
        # Deliberately "qwen2" (not "qwen" as in step 3): the paper's generation harness used this test,
        # which is False for hkust-nlp/Qwen-2.5-*-SimpleRL-Zoo and Qwen/Qwen3.5-0.8B.
        stop_token_ids=[151645, 151643] if "qwen2" in args.model.lower() else None,
    )
    start_time = time.time()
    outputs = sorted(llm.generate(prompts, sampling), key=lambda x: int(x.request_id))
    responses = []
    for prompt, output in zip(prompts, outputs):
        response = (prompt + output.outputs[0].text.rstrip()).split(prompt)[-1].strip()
        for stop_word in STOP_WORDS:
            if stop_word in response:
                response = response.split(stop_word)[0].strip()
        responses.append(response)
    preds = [predict(r) for r in responses]
    time_use = time.time() - start_time

    # 3. Attach response and prediction, re-normalise the reference, grade.
    for record, response, pred in zip(records, responses, preds):
        pred = clean_multiple_choice(pred, record["gt"], response)
        record.update({"code": [response], "pred": [pred], "report": [None]})
    records = sorted({r["idx"]: r for r in records}.values(), key=lambda r: r["idx"])
    for record in records:
        record["gt_cot"], record["gt"] = parse_ground_truth(record, args.benchmark)
    scores, n_timeouts = grade_predictions([(r["pred"][0], r["gt"]) for r in records])
    for record, score in zip(records, scores):
        record["score"] = [score]

    summary = accuracy_summary(records, n_timeouts)
    summary["time_use_in_second"] = time_use
    summary["time_use_in_minute"] = f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    return records, summary


def main():
    args = parse_args()
    # Same seeding as the paper's harness (vLLM sampling uses its own engine seed).
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    out_file = os.path.join(args.output_dir, args.benchmark, output_filename(
        args.prompt_type, args.max_samples, args.seed, args.temperature))
    metrics_file = out_file.replace(".jsonl", "_metrics.json")
    if os.path.exists(metrics_file):
        print(f"Skipping: {out_file} is already graded.")
        return

    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size or torch.cuda.device_count(),
        gpu_memory_utilization=0.75,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True) if args.apply_chat_template else None

    records, summary = generate(llm, tokenizer, args)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    write_jsonl(records, out_file)
    with open(metrics_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"{args.benchmark}: acc={summary['acc']:.1f} -> {out_file}")


if __name__ == "__main__":
    main()
