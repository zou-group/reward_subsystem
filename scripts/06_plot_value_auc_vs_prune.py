#!/usr/bin/env python
"""Step 6 - Value probe: reward-prediction AUC versus pruning ratio, and its figure.

``curve``  For each pruning ratio (0, 0.1, ..., 0.9, 0.92, 0.94, 0.96, 0.98, 0.99) the
           least important input neurons of the value probe are zeroed and every
           validation problem of step 2 is scored by V(s0): the probe's value at the
           last token of the *question-only* text (no response). The LLM is run once;
           the pruned probes are applied to the same hidden states. AUC = ROC-AUC of
           the scores against the binary (string-equality) reward; error bar = std of
           1000 bootstrap AUCs.

           The validation problems are read from the checkpoint (``split.valid_idx``,
           in split order). Checkpoints written by the paper's original code carry no
           split; for them the split is re-created with the paper's settings
           (``RandomState(0).permutation``, 80/20), as the original evaluation did.
           --outputs_jsonl, --benchmark, --model, --layer and --apply_chat_template
           must match step 2.

``plot``   One ``--panel`` per panel: a title, then ``LABEL=auc_curve.json`` entries
           (e.g. one per layer). A .png is written next to the .pdf.

Examples
    python scripts/06_plot_value_auc_vs_prune.py curve --model hkust-nlp/Qwen-2.5-14B-SimpleRL-Zoo \
        --outputs_jsonl <ARTIFACT_DIR>/outputs/output_14B_simplerl_16000tk/math500/test_qwen-boxed_-1_seed0_t1.0_s0_e-1.jsonl \
        --benchmark math500 --apply_chat_template --layer 2 \
        --probe <ARTIFACT_DIR>/value_probes/14B_simplerl_math500/layer_2/value_probe_best.pt \
        --output_json <ARTIFACT_DIR>/value_auc/14B_simplerl_math500_layer2.json

    python scripts/06_plot_value_auc_vs_prune.py plot --output <ARTIFACT_DIR>/figures/value_auc_14B.pdf \
        --panel "MATH 500" "Layer 2=<ARTIFACT_DIR>/value_auc/14B_simplerl_math500_layer2.json" \
                           "Layer 3=<ARTIFACT_DIR>/value_auc/14B_simplerl_math500_layer3.json"
"""

import argparse
import json
import os

import torch

from reward_subsystem.models import load_frozen_lm, set_seed
from reward_subsystem.plotting import parse_panels, plot_figure
from reward_subsystem.probes import load_value_probe
from reward_subsystem.pruning import VALUE_PRUNE_RATIOS, pruned
from reward_subsystem.value_probe import (
    auc_with_bootstrap_std, extract_hidden_states, load_labeled_problems, score_last_token,
    train_valid_split,
)


# Split of checkpoints written by the paper's original code (no ``split`` entry).
PAPER_SPLIT = {"train_ratio": 0.8, "seed": 0}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("curve", help="AUC versus pruning ratio of one value probe")
    c.add_argument("--probe", required=True, help="value_probe_best.pt from step 2")
    c.add_argument("--outputs_jsonl", required=True)
    c.add_argument("--benchmark", default="math500")
    c.add_argument("--data_dir", default="./data")
    c.add_argument("--model", required=True)
    c.add_argument("--layer", type=int, required=True)
    c.add_argument("--apply_chat_template", action="store_true")
    c.add_argument("--output_json", required=True)
    c.add_argument("--batch_size", type=int, default=4)
    c.add_argument("--max_length", type=int, default=2048)
    c.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    c.add_argument("--n_bootstrap", type=int, default=1000)
    c.add_argument("--bootstrap_seed", type=int, default=0)

    f = sub.add_parser("plot", help="figure from AUC curve files")
    f.add_argument("--panel", nargs="+", action="append", required=True, metavar="TITLE LABEL=PATH",
                   help="Panel title, then one LABEL=auc_curve.json per curve.")
    f.add_argument("--output", required=True, help="Figure path (.pdf); a .png is written next to it.")
    return p.parse_args()


def validation_positions(problems, split: dict) -> list:
    """Positions in ``problems`` of the step-2 validation problems, in split order."""
    if "valid_idx" not in split:
        _, positions = train_valid_split(len(problems), split["train_ratio"], split["seed"])
        return positions
    position_of = {p.idx: pos for pos, p in enumerate(problems)}
    missing = [i for i in split["valid_idx"] if i not in position_of]
    if missing:
        raise SystemExit(f"{len(missing)} validation problems of the probe are not in --outputs_jsonl "
                         f"(e.g. idx {missing[:5]}); use the outputs file the probe was trained on.")
    return [position_of[i] for i in split["valid_idx"]]


def curve(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe, ckpt = load_value_probe(args.probe, device)
    split = ckpt.get("split", PAPER_SPLIT)
    set_seed(split["seed"])

    problems = load_labeled_problems(args.outputs_jsonl, args.benchmark, args.data_dir)
    positions = validation_positions(problems, split)
    print(f"[data] scoring {len(positions)} validation problems of {len(problems)}")

    model, tokenizer = load_frozen_lm(args.model, args.dtype)
    batches = extract_hidden_states(
        model, tokenizer, problems, positions, layer=args.layer,
        apply_chat_template=args.apply_chat_template, with_response=False,
        batch_size=args.batch_size, max_length=args.max_length, device=device)
    del model

    ratios = sorted(VALUE_PRUNE_RATIOS)
    scores, aucs, stds = [], [], []
    for ratio in ratios:
        with pruned(probe.first_linear, ratio):
            idx, labels, ratio_scores = score_last_token(probe, batches, device)
        auc, std = auc_with_bootstrap_std(labels, ratio_scores, args.bootstrap_seed, args.n_bootstrap)
        scores.append(ratio_scores)
        aucs.append(auc)
        stds.append(std)
        print(f"  prune {ratio:.2f} -> AUC {auc:.4f} +- {std:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({
            "probe": args.probe, "model": args.model, "benchmark": args.benchmark, "layer": args.layer,
            "n_problems": len(labels), "prune_ratios": ratios, "auc": aucs, "auc_std": stds,
            "idx": idx, "labels": labels, "scores": scores,
        }, f)
    print(f"[done] {len(idx)} problems x {len(ratios)} ratios -> {args.output_json}")


def main():
    args = parse_args()
    if args.command == "curve":
        curve(args)
    else:
        plot_figure(parse_panels(args.panel), args.output, kind="value_auc")
        print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
