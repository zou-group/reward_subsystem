"""Benchmarks: loading problems, parsing questions / reference answers, building prompts.

Supported benchmarks (files under ``<data_dir>``):

    math500       math500/test.jsonl
    gsm8k         gsm8k/test.jsonl
    arc           arc/arc.json          (ARC-Challenge, verl-style records)

Every problem gets an integer ``idx`` (its line number if the file has none);
``idx`` is the key that ties prompts, model outputs, labels and splits together.
"""

import json
from typing import Dict, List

from reward_subsystem.answer_parsing import extract_answer, strip_string


SUPPORTED_BENCHMARKS = ("math500", "gsm8k", "arc")

_DATA_FILES = {
    "math500": "math500/test.jsonl",
    "gsm8k": "gsm8k/test.jsonl",
    "arc": "arc/arc.json",
}

# Zero-shot prompt templates (``{input}`` is the question):
#   qwen-boxed   generation for Qwen models (step 1), and the dopamine pipeline (steps 3-4)
#   llama-boxed  generation for non-Qwen models, combined with a chat template (step 1)
#   cot          value-probe text (steps 2 and 6)
PROMPT_TEMPLATES = {
    "qwen-boxed": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    "llama-boxed": "{input}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    "cot": "Question: {input}\nAnswer: ",
}


def read_jsonl(path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_benchmark(name: str, data_dir: str) -> List[dict]:
    """Read a benchmark file, add ``idx`` if missing, and sort by ``idx``."""
    if name not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark '{name}'. Supported: {SUPPORTED_BENCHMARKS}")
    examples = read_jsonl(f"{data_dir}/{_DATA_FILES[name]}")
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]
    return sorted(examples, key=lambda x: x["idx"])


def parse_ground_truth(example: dict, data_name: str):
    """Return ``(reference_solution, reference_answer)`` for one problem.

    If the record already carries ``gt_cot``/``gt`` (a saved model-output record),
    the stored answer is re-normalised, exactly as the evaluation harness does.
    """
    if "gt_cot" in example and "gt" in example:
        return example["gt_cot"], strip_string(example["gt"])

    if data_name == "math500":
        gt_cot = example["solution"]
        gt_ans = extract_answer(gt_cot)
    elif data_name == "gsm8k":
        gt_cot, gt_ans = example["answer"].split("####")
    elif data_name == "arc":
        gt_cot, gt_ans = None, example["reward_model"]["ground_truth"]
    else:
        raise NotImplementedError(f"`{data_name}`")

    return str(gt_cot).strip(), strip_string(gt_ans)


def parse_question(example: dict, data_name: str) -> str:
    """Return the question text of one problem."""
    question = ""
    if data_name == "arc":
        prompts = example["prompt"][1]["content"].split("answering.\n\n")[-1].strip()
        question = (
            prompts.replace("\n\nA)", "\nAnswer Choices: (A)")
            .replace("\nB)", " (B)")
            .replace("\nC)", " (C)")
            .replace("\nD)", " (D)")
            .replace("\nE)", " (E)")
        )
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    # Yes/No and True/False questions get an explicit hint.
    _, gt_ans = parse_ground_truth(example, data_name)
    if isinstance(gt_ans, str):
        gt_lower = gt_ans.lower()
        if gt_lower in ["true", "false"]:
            question += " (True or False)"
        if gt_lower in ["yes", "no"]:
            question += " (Yes or No)"
    return question.strip()


def construct_prompt(question: str, prompt_type: str) -> str:
    """Zero-shot prompt for ``question`` (surrounding spaces stripped, as in the harness)."""
    return PROMPT_TEMPLATES[prompt_type].format(input=question).strip(" ")


def build_prompts(data_name: str, data_dir: str, prompt_type: str) -> List[Dict]:
    """All problems of a benchmark as ``{idx, question, prompt, gt}``, sorted by ``idx``.

    Problems with an empty question are skipped.
    """
    samples = []
    for example in load_benchmark(data_name, data_dir):
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        _, gt_ans = parse_ground_truth(example, data_name)
        samples.append({
            "idx": example["idx"],
            "question": example["question"],
            "prompt": construct_prompt(example["question"], prompt_type),
            "gt": gt_ans,
        })
    samples.sort(key=lambda x: x["idx"])
    return samples
