#!/usr/bin/env python
"""Step 7 - Dopamine probe: Spearman correlation versus pruning ratio, and its figure.

``curve``  Spearman rho between the pruned dopamine probe's prediction and the TD
           error delta on the held-out problems of step 5 (``test_problem_idxs.json``,
           same |delta| > threshold filter), over the 52-point ratio grid
           ``pruning.DOPAMINE_PRUNE_RATIOS``. Runs on CPU in seconds.

``plot``   One ``--panel`` per panel: a title, then ``LABEL=spearman_curve.json``
           entries (e.g. one per layer). The figure shows the ratios
           0, 0.1, ..., 0.9, 0.95. A .png is written next to the .pdf.

Examples
    python scripts/07_plot_dopamine_spearman_vs_prune.py curve \
        --result_dir <ARTIFACT_DIR>/dopamine/td_dopamine_7B_simplerl_math500 \
        --probe_dir <ARTIFACT_DIR>/dopamine_probes/7B_simplerl_math500/layer20 \
        --output_json <ARTIFACT_DIR>/dopamine_probes/7B_simplerl_math500/layer20/spearman_curve.json

    python scripts/07_plot_dopamine_spearman_vs_prune.py plot --output <ARTIFACT_DIR>/figures/dopamine_spearman.pdf \
        --panel "Qwen2.5-7B-RL" "Layer 20=<ARTIFACT_DIR>/dopamine_probes/7B_simplerl_math500/layer20/spearman_curve.json" \
                                "Layer 21=<ARTIFACT_DIR>/dopamine_probes/7B_simplerl_math500/layer21/spearman_curve.json"
"""

import argparse
import json
import os

import torch
from scipy.stats import spearmanr

from reward_subsystem.dopamine import CACHE_RELPATH, load_dopamine_dataset, rows_of_problems
from reward_subsystem.plotting import parse_panels, plot_figure
from reward_subsystem.probes import load_dopamine_probe
from reward_subsystem.pruning import DOPAMINE_PRUNE_RATIOS, pruned


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("curve", help="Spearman versus pruning ratio of one dopamine probe")
    c.add_argument("--result_dir", required=True, help="directory of steps 3-4")
    c.add_argument("--probe_dir", required=True, help="output directory of step 5")
    c.add_argument("--output_json", required=True)

    f = sub.add_parser("plot", help="figure from Spearman curve files")
    f.add_argument("--panel", nargs="+", action="append", required=True, metavar="TITLE LABEL=PATH",
                   help="Panel title, then one LABEL=spearman_curve.json per curve.")
    f.add_argument("--output", required=True, help="Figure path (.pdf); a .png is written next to it.")
    return p.parse_args()


def curve(args):
    probe, ckpt = load_dopamine_probe(os.path.join(args.probe_dir, "dopamine_probe.pt"), "cpu")
    with open(os.path.join(args.probe_dir, "test_problem_idxs.json")) as f:
        test_problems = json.load(f)
    td, X, sample_idxs, _ = load_dopamine_dataset(
        os.path.join(args.result_dir, CACHE_RELPATH), ckpt["layer"], ckpt["td_threshold"])
    test_rows = rows_of_problems(sample_idxs, test_problems)
    X_test = torch.tensor(X[test_rows], dtype=torch.float32)
    y_test = td[test_rows]

    rhos = []
    for ratio in DOPAMINE_PRUNE_RATIOS:
        with pruned(probe.first_linear, ratio), torch.no_grad():
            pred = probe(X_test).numpy()
        rhos.append(spearmanr(y_test, pred)[0])
    print(f"  layer {ckpt['layer']}: Spearman {rhos[0]:.4f} unpruned, "
          f"{rhos[DOPAMINE_PRUNE_RATIOS.index(0.9)]:.4f} at 90% pruning")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"probe_dir": args.probe_dir, "layer": ckpt["layer"], "n_test_paragraphs": len(y_test),
                   "prune_ratios": DOPAMINE_PRUNE_RATIOS, "spearman": rhos}, f, indent=2)
    print(f"[done] -> {args.output_json}")


def main():
    args = parse_args()
    if args.command == "curve":
        curve(args)
    else:
        plot_figure(parse_panels(args.panel), args.output, kind="dopamine_spearman")
        print(f"[done] -> {args.output}")


if __name__ == "__main__":
    main()
