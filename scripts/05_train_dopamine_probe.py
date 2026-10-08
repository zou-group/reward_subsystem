#!/usr/bin/env python
"""Step 5 - Train the dopamine probe of one layer and rank dopamine neurons.

Keeps the paragraphs with |delta| > --td_threshold, splits them by problem
(80/20, ``RandomState(seed)`` shuffle of problem ids) and fits an MLP
(hidden_dim -> 32 -> ReLU -> 1) to the TD errors. Dopamine neurons are the
input dimensions with the largest L1 norm of their first-layer weight column.

Inputs
    --result_dir   directory of steps 3-4 (reads .cache/hidden_states_all_layers.npz)
    --layer        decoder block index, 0-based (= HF hidden_states[layer + 1])

Outputs (in --output_dir)
    dopamine_probe.pt        {model_state_dict, input_dim, mlp_hidden, layer, td_threshold, pearson_r, spearman_rho}
    dopamine_neurons.npz     {importance, ranked_indices, n_layers=1, hidden_dim, layer_offset=layer}
    test_problem_idxs.json   problem ids of the held-out split (read by step 7)

Note: mini-batches are drawn with ``torch.randperm`` on the training device;
CPU and GPU runs therefore give different (equally valid) probes. The paper used a GPU.

Example
    python scripts/05_train_dopamine_probe.py --layer 20 \
        --result_dir <ARTIFACT_DIR>/dopamine/td_dopamine_7B_simplerl_math500 \
        --output_dir <ARTIFACT_DIR>/dopamine_probes/7B_simplerl_math500/layer20
"""

import argparse
import json
import os

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from reward_subsystem.dopamine import (
    CACHE_RELPATH, load_dopamine_dataset, problem_level_split, train_dopamine_probe,
)
from reward_subsystem.models import set_seed
from reward_subsystem.pruning import neuron_importance


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result_dir", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--td_threshold", type=float, default=0.3)
    p.add_argument("--mlp_hidden", type=int, default=32)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != torch.device(args.device).type:
        print(f"[warning] --device {args.device} is not available; training on CPU. Mini-batches are drawn "
              f"on the training device, so the probe differs from a GPU run (the paper used a GPU).")

    td, X, sample_idxs, hidden_dim = load_dopamine_dataset(
        os.path.join(args.result_dir, CACHE_RELPATH), args.layer, args.td_threshold)
    if len(td) < 20:
        raise SystemExit(f"Only {len(td)} paragraphs pass the TD threshold; too few to train a probe.")
    train_rows, test_rows, test_problems = problem_level_split(sample_idxs, args.train_ratio, args.seed)
    print(f"[data] layer {args.layer}: {len(td)} paragraphs with |delta| > {args.td_threshold}; "
          f"train {len(train_rows)} / test {len(test_rows)} paragraphs, {len(test_problems)} test problems")
    with open(os.path.join(args.output_dir, "test_problem_idxs.json"), "w") as f:
        json.dump(test_problems, f)

    X_test, y_test = X[test_rows], td[test_rows]
    probe, best_state = train_dopamine_probe(
        X[train_rows], td[train_rows], X_test, y_test, mlp_hidden=args.mlp_hidden,
        num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, device=device)

    with torch.no_grad():
        pred_test = probe(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()
    r, rho = pearsonr(y_test, pred_test)[0], spearmanr(y_test, pred_test)[0]
    print(f"[result] test Pearson r = {r:.4f}, Spearman rho = {rho:.4f}")

    importance = neuron_importance(probe.first_linear)
    ranked = np.argsort(-importance)
    print("[result] top-10 dopamine neurons (dim): " + ", ".join(str(int(d)) for d in ranked[:10]))

    torch.save({
        "model_state_dict": best_state,
        "input_dim": X.shape[1],
        "mlp_hidden": args.mlp_hidden,
        "layer": args.layer,
        "td_threshold": args.td_threshold,
        "pearson_r": r,
        "spearman_rho": rho,
    }, os.path.join(args.output_dir, "dopamine_probe.pt"))
    np.savez(os.path.join(args.output_dir, "dopamine_neurons.npz"),
             importance=importance, ranked_indices=ranked,
             n_layers=1, hidden_dim=hidden_dim, layer_offset=args.layer)
    print(f"[done] -> {args.output_dir}")


if __name__ == "__main__":
    main()
