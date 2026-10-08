#!/usr/bin/env python
"""Step 4 - Paragraph-level hidden-state features of all layers (dopamine pipeline, part 2).

Run after step 3 has finished: only problems with a ``sample_results/<idx>.json``
are used (the count is printed). For each of them, one forward pass over
prompt + response; the hidden states of every decoder block are z-scored over
the response tokens and averaged within each paragraph. Rows follow problem
``idx`` order, then paragraph order.

Inputs
    --result_dir      the directory of step 3 (reads sample_results/<idx>.json)
    --outputs_jsonl, --benchmark, --model   same as step 3

Output
    <result_dir>/.cache/hidden_states_all_layers.npz
        td_errors     (N,)                     delta of each paragraph
        values        (N,)                     V(t+1) at the end of each paragraph
        hidden_states (N, n_layers*hidden_dim) float32; column f = block f // hidden_dim, dim f % hidden_dim
        sample_idxs   (N,)                     problem idx of each paragraph
        n_layers, hidden_dim

Example
    python scripts/04_dopamine_features.py --model hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo \
        --outputs_jsonl <ARTIFACT_DIR>/outputs/output_7B_simplerl_16000tk/math500/test_qwen-boxed_-1_seed0_t1.0_s0_e-1.jsonl \
        --benchmark math500 --result_dir <ARTIFACT_DIR>/dopamine/td_dopamine_7B_simplerl_math500
"""

import argparse
import json
import os

import numpy as np

from reward_subsystem.dopamine import CACHE_RELPATH, load_rollout_problems, paragraph_features
from reward_subsystem.models import load_frozen_lm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs_jsonl", required=True)
    p.add_argument("--benchmark", default="math500")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--model", required=True)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--max_samples", type=int, default=-1, help="Use only the first N problems (-1 = all).")
    p.add_argument("--max_length", type=int, default=4096)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()


def main():
    args = parse_args()
    cache_path = os.path.join(args.result_dir, CACHE_RELPATH)
    problems = load_rollout_problems(args.outputs_jsonl, args.benchmark, args.data_dir, args.max_samples)
    results_dir = os.path.join(args.result_dir, "sample_results")
    results = []
    for problem in problems:
        path = os.path.join(results_dir, f"{problem.idx}.json")
        if os.path.exists(path):
            with open(path) as f:
                results.append((problem, json.load(f)))
    print(f"[data] {len(results)} problems with TD errors (of {len(problems)})", flush=True)
    if len(results) < len(problems):
        print(f"[warning] {len(problems) - len(results)} problems have no sample_results/<idx>.json; "
              f"is step 3 finished?", flush=True)

    model, tokenizer = load_frozen_lm(args.model, args.dtype)
    td_errors, values, features, sample_idxs = [], [], [], []
    for i, (problem, result) in enumerate(results):
        paragraph_feats, n_layers, hidden_dim = paragraph_features(
            model, tokenizer, problem.prompt, result["sentences"], args.max_length)
        for t, feat in enumerate(paragraph_feats):
            td_errors.append(result["td_errors"][t])
            values.append(result["values"][t + 1])
            features.append(feat)
            sample_idxs.append(problem.idx)
        if (i + 1) % 50 == 0:
            print(f"[features] {i + 1}/{len(results)}", flush=True)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, td_errors=np.array(td_errors), values=np.array(values),
             hidden_states=np.stack(features, axis=0), sample_idxs=np.array(sample_idxs, dtype=np.int64),
             n_layers=n_layers, hidden_dim=hidden_dim)
    print(f"[done] {len(td_errors)} paragraphs, {n_layers} layers x {hidden_dim} dims -> {cache_path}")


if __name__ == "__main__":
    main()
