#!/usr/bin/env python
"""Step 2 - Train the value probe (value neurons) with a token-level TD loss.

Inputs
    --outputs_jsonl   model outputs from step 1; label = string-equality reward
    --benchmark       benchmark of that file
    --model           the HF model that produced the outputs (frozen)
    --layer           index into HF ``hidden_states`` (0 = embeddings, L = output of block L-1)

The probe text is ``Question: <q>\\nAnswer: <response>`` (the ``cot`` prompt), plain
or, with --apply_chat_template, wrapped in the math chat template.

Outputs (in --output_dir)
    value_probe_best.pt    probe with the lowest validation TD loss (used by step 6)
    value_probe_final.pt   probe after the last epoch
    Both store the split (``split = {train_ratio, seed, valid_idx}``) next to the weights.

Problems are split 80/20 with ``RandomState(seed).permutation``; step 6 reads
the validation problem ids from the checkpoint. Training batches follow a fixed
order (no shuffling). The LLM is run once and its hidden states are reused
in every epoch (the paper's code recomputed them each epoch; the frozen model
gives the same tensors).

Memory: the hidden states of one layer are cached in CPU RAM in the model dtype,
up to problems x max_length x hidden_size x 2 bytes (bf16), i.e. about 7 GB for
7B and 10.5 GB for 14B on MATH-500.

Example
    python scripts/02_train_value_probe.py --model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
        --outputs_jsonl <ARTIFACT_DIR>/outputs/output_14B_simplerl_16000tk/math500/test_qwen-boxed_-1_seed0_t1.0_s0_e-1.jsonl \
        --benchmark math500 --apply_chat_template --layer 2 \
        --output_dir <ARTIFACT_DIR>/value_probes/14B_simplerl_math500/layer_2
"""

import argparse
import os

import torch

from reward_subsystem.models import hidden_size, load_frozen_lm, set_seed
from reward_subsystem.probes import MLPProbe, save_value_probe
from reward_subsystem.value_probe import (
    extract_hidden_states, load_labeled_problems, run_td_epoch, train_valid_split,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs_jsonl", required=True)
    p.add_argument("--benchmark", default="math500")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--apply_chat_template", action="store_true",
                   help="Wrap prompt + response in the math chat template (see README for per-figure settings). "
                        "Use the same setting in step 6.")
    p.add_argument("--mlp_hidden", type=int, default=1024)
    p.add_argument("--num_epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problems = load_labeled_problems(args.outputs_jsonl, args.benchmark, args.data_dir)
    train_pos, valid_pos = train_valid_split(len(problems), args.train_ratio, args.seed)
    print(f"[data] {len(problems)} problems ({sum(p.reward for p in problems):.0f} correct), "
          f"train {len(train_pos)} / valid {len(valid_pos)}")

    model, tokenizer = load_frozen_lm(args.model, args.dtype)
    hidden = hidden_size(model)
    probe = MLPProbe(hidden, args.mlp_hidden).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    extract = dict(model=model, tokenizer=tokenizer, problems=problems, layer=args.layer,
                   apply_chat_template=args.apply_chat_template, with_response=True,
                   batch_size=args.batch_size, max_length=args.max_length, device=device)
    train_batches = extract_hidden_states(positions=train_pos, **extract)
    valid_batches = extract_hidden_states(positions=valid_pos, **extract)
    del model

    split = dict(train_ratio=args.train_ratio, seed=args.seed,
                 valid_idx=[int(problems[i].idx) for i in valid_pos])
    ckpt_info = dict(hidden_size=hidden, mlp_hidden=args.mlp_hidden, gamma=args.gamma, split=split)
    best_val_loss, best_epoch = float("inf"), -1
    for epoch in range(1, args.num_epochs + 1):
        train_loss = run_td_epoch(probe, train_batches, args.gamma, device, optimizer)
        valid_loss = run_td_epoch(probe, valid_batches, args.gamma, device)
        print(f"[epoch {epoch}/{args.num_epochs}] train TD loss {train_loss:.6f} | valid TD loss {valid_loss:.6f}",
              flush=True)
        if valid_loss < best_val_loss:
            best_val_loss, best_epoch = valid_loss, epoch
            save_value_probe(os.path.join(args.output_dir, "value_probe_best.pt"), probe,
                             epoch=epoch, val_loss=valid_loss, **ckpt_info)

    save_value_probe(os.path.join(args.output_dir, "value_probe_final.pt"), probe,
                     epoch=args.num_epochs, val_loss=best_val_loss, **ckpt_info)
    print(f"[done] best epoch {best_epoch}, best valid TD loss {best_val_loss:.6f} -> {args.output_dir}")


if __name__ == "__main__":
    main()
