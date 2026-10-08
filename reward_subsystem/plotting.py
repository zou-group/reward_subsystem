"""Paper-style figures of the two pruning curves.

A figure is a row of panels (e.g. one per benchmark or model); each panel overlays
several curves (e.g. one per layer). Curve files are written by steps 6 and 7:

    value AUC curve         {"prune_ratios": [...], "auc": [...], "auc_std": [...], ...}
    dopamine Spearman curve {"prune_ratios": [...], "spearman": [...], ...}
"""

import json
import os
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = ["#4A6670", "#6A9EC0", "#A8D0E6", "#2F4550", "#93B7CF"]
MARKERS = ["o", "^", "s", "D", "v"]
# The dopamine figures show this subset of the 52-point ratio grid.
DOPAMINE_PLOT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Liberation Sans", "DejaVu Sans"]


def load_curve(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def parse_panels(panel_args) -> List[Tuple[str, List[Tuple[str, dict]]]]:
    """``[[TITLE, "LABEL=curve.json", ...], ...]`` (argparse ``--panel``) -> loaded panels."""
    panels = []
    for title, *entries in panel_args:
        curves = []
        for entry in entries:
            label, path = entry.split("=", 1)
            curves.append((label, load_curve(path)))
        panels.append((title, curves))
    return panels


def _plot_value_auc(ax, label, curve, color, marker):
    ratios, auc = np.array(curve["prune_ratios"]), np.array(curve["auc"], dtype=float)
    std = np.array(curve["auc_std"], dtype=float)
    ax.plot(ratios, auc, marker=marker, linestyle="-", color=color, linewidth=2, markersize=6, label=label)
    if np.any(std > 0):
        ax.fill_between(ratios, auc - std, auc + std, color=color, alpha=0.12, linewidth=0)


def _plot_dopamine_spearman(ax, label, curve, color, marker):
    ratios, rho = np.array(curve["prune_ratios"]), np.array(curve["spearman"], dtype=float)
    keep = np.zeros(len(ratios), dtype=bool)
    for g in DOPAMINE_PLOT_GRID:
        keep |= np.isclose(ratios, g, atol=1e-3)
    ax.plot(ratios[keep], rho[keep], marker=marker, linestyle="-", color=color,
            linewidth=2, markersize=6, label=label)


def _style_axis(ax, kind: str, title: str) -> None:
    ax.set_xlim(-0.04, 1.04)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Prune Ratio", fontsize=14)
    if kind == "value_auc":
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel("AUC", fontsize=14)
    else:
        ax.set_ylim(-0.02, 0.42)
        ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
        ax.set_ylabel(r"Spearman $\rho$", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower left", fontsize=11, frameon=True)


def plot_figure(panels: Sequence[Tuple[str, Sequence[Tuple[str, dict]]]], output: str, kind: str) -> None:
    """Save a 1 x len(panels) figure as ``output`` (PDF at 300 dpi) plus a PNG next to it.

    ``kind`` is ``"value_auc"`` (AUC with a +-std band) or ``"dopamine_spearman"``.
    """
    plot_curve = {"value_auc": _plot_value_auc, "dopamine_spearman": _plot_dopamine_spearman}[kind]
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.4), squeeze=False)
    for ax, (title, curves) in zip(axes[0], panels):
        for (label, curve), color, marker in zip(curves, PALETTE, MARKERS):
            plot_curve(ax, label, curve, color, marker)
        _style_axis(ax, kind, title)
    plt.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.rsplit(".", 1)[0] + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
