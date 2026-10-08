#!/usr/bin/env python
"""Step 3 - Monte-Carlo values and TD errors at paragraph boundaries (dopamine pipeline, part 1).

For every problem, the response from step 1 is split into paragraphs. From each
boundary t (prompt + first t paragraphs) we sample --n_rollouts continuations
with vLLM, grade them with ``math_equal``, and set V(t) = fraction correct;
V(n) = the response's string-equality reward. TD error: delta(t) = gamma*V(t+1) - V(t).

The prompt is rebuilt with the ``qwen-boxed`` template and wrapped in the math
chat template (so the chat template is applied twice, as in the paper).

Inputs
    --outputs_jsonl   model outputs from step 1
    --benchmark, --model

Outputs (in --result_dir; resumable, finished problems are skipped)
    sample_results/<idx>.json                  {idx, reward, sentences, values, td_errors}
    sample_results/_raw_completions/<idx>.json  raw rollout texts per boundary

Example
    python scripts/03_dopamine_td_errors.py --model hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo \
        --outputs_jsonl <ARTIFACT_DIR>/outputs/output_7B_simplerl_16000tk/math500/test_qwen-boxed_-1_seed0_t1.0_s0_e-1.jsonl \
        --benchmark math500 --result_dir <ARTIFACT_DIR>/dopamine/td_dopamine_7B_simplerl_math500
"""

import os

# The paper ran rollouts on vLLM's V0 engine (vllm==0.10.2). Set before importing vLLM.
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json

from transformers import AutoTokenizer

from reward_subsystem.dopamine import (
    NoDaemonPool, boundary_prefixes, load_rollout_problems, td_errors_from_rollouts,
)
from reward_subsystem.models import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs_jsonl", required=True)
    p.add_argument("--benchmark", default="math500")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--model", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--n_rollouts", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--max_samples", type=int, default=-1, help="Use only the first N problems (-1 = all).")
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False)


def generate_rollouts(problems, tokenizer, raw_dir, args):
    """Sample rollouts for every boundary of ``problems`` (one vLLM call) and save the raw texts."""
    import torch
    from vllm import LLM, SamplingParams

    prefixes, keys = [], []
    for problem in problems:
        for t, prefix in enumerate(boundary_prefixes(tokenizer, problem)):
            prefixes.append(prefix)
            keys.append((problem.idx, t))
    print(f"[rollouts] {len(prefixes)} prefixes x {args.n_rollouts} rollouts", flush=True)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p if args.temperature > 0 else 1.0,
        max_tokens=args.max_new_tokens,
        n=args.n_rollouts,
        stop=["</s>", "<|im_end|>", "<|endoftext|>"],
        # Deliberately "qwen" (not "qwen2" as in step 1): the paper's rollout code used this test.
        stop_token_ids=[151645, 151643] if "qwen" in args.model.lower() else None,
    )
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    outputs = llm.generate(prefixes, sampling)
    del llm
    torch.cuda.empty_cache()

    completions = {}
    for (idx, t), output in zip(keys, outputs):
        completions.setdefault(idx, {})[str(t)] = [c.text for c in output.outputs]
    os.makedirs(raw_dir, exist_ok=True)
    for idx, by_boundary in completions.items():
        raw_path = os.path.join(raw_dir, f"{idx}.json")
        if not os.path.exists(raw_path):
            write_json({"idx": idx, "completions": by_boundary}, raw_path)


def main():
    args = parse_args()
    set_seed(args.seed)
    results_dir = os.path.join(args.result_dir, "sample_results")
    raw_dir = os.path.join(results_dir, "_raw_completions")
    os.makedirs(results_dir, exist_ok=True)

    problems = load_rollout_problems(args.outputs_jsonl, args.benchmark, args.data_dir, args.max_samples)
    done = {p.idx for p in problems if os.path.exists(os.path.join(results_dir, f"{p.idx}.json"))}
    to_grade = [p for p in problems if p.idx not in done]
    to_generate = [p for p in to_grade if not os.path.exists(os.path.join(raw_dir, f"{p.idx}.json"))]
    print(f"[data] {len(problems)} problems: {len(done)} done, {len(to_generate)} need rollouts, "
          f"{len(to_grade) - len(to_generate)} need grading only", flush=True)

    if to_generate:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        generate_rollouts(to_generate, tokenizer, raw_dir, args)

    with NoDaemonPool(min(os.cpu_count() or 1, 32)) as pool:
        for i, problem in enumerate(to_grade):
            completions = read_json(os.path.join(raw_dir, f"{problem.idx}.json"))["completions"]
            result = td_errors_from_rollouts(problem, completions, args.gamma, pool)
            write_json(result, os.path.join(results_dir, f"{problem.idx}.json"))
            if (i + 1) % 20 == 0 or i + 1 == len(to_grade):
                print(f"[grading] {i + 1}/{len(to_grade)}", flush=True)


if __name__ == "__main__":
    main()
