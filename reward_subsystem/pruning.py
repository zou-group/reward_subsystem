"""Neuron ranking and structured pruning of a probe's first layer.

A probe's first linear layer has weight ``W`` of shape ``(hidden_dim, input_dim)``.
Input neuron ``j`` (one hidden-state coordinate of the LLM) is scored by the
magnitude of its column ``W[:, j]``. Pruning ratio ``p`` zeroes the
``int(p * input_dim)`` lowest-scoring columns. The neurons that survive heavy
pruning are the value neurons (value probe) or dopamine neurons (dopamine probe).
"""

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn


# Ratio grid of the value-probe AUC curves (step 6).
VALUE_PRUNE_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]

# Ratio grid of the dopamine-probe Spearman curves (step 7). The raw ``np.arange`` floats are kept on
# purpose (e.g. 0.7000000000000001): ``int(p * input_dim)`` is evaluated on them.
DOPAMINE_PRUNE_RATIOS = sorted(set(np.arange(0, 1.0, 0.02).tolist() + [0.95, 0.98, 0.99]))


def neuron_importance(first_linear: nn.Linear) -> np.ndarray:
    """L1 norm of each input column of the first layer, shape ``(input_dim,)``.

    This is the ranking saved as ``dopamine_neurons.npz`` (``np.argsort(-importance)``).
    ``prune_input_neurons`` ranks by the column *mean* ``|W|`` with ``torch.topk``
    instead. Both orders agree except among columns with tied scores, where the two
    sorts may break ties differently; both are the paper's behaviour and kept as is.
    """
    return first_linear.weight.data.abs().sum(dim=0).cpu().numpy()


def prune_input_neurons(first_linear: nn.Linear, orig_weight: torch.Tensor, prune_ratio: float):
    """Zero the ``int(prune_ratio * input_dim)`` least important input columns.

    Importance is the mean ``|W[:, j]|`` (same ranking as the L1 norm). Always
    starts from ``orig_weight``, so successive calls do not compound.
    Returns ``(num_pruned, input_dim)``.
    """
    with torch.no_grad():
        w = orig_weight.clone()
        _, in_features = w.shape
        num_prune = int(prune_ratio * in_features)
        if num_prune <= 0:
            first_linear.weight.data.copy_(w)
            return 0, in_features
        col_scores = w.abs().mean(dim=0)
        _, prune_idx = torch.topk(col_scores, k=num_prune, largest=False)
        w[:, prune_idx] = 0.0
        first_linear.weight.data.copy_(w)
        return num_prune, in_features


@contextmanager
def pruned(first_linear: nn.Linear, prune_ratio: float):
    """Temporarily prune ``first_linear`` to ``prune_ratio``; restore the weights on exit."""
    orig_weight = first_linear.weight.data.clone()
    prune_input_neurons(first_linear, orig_weight, prune_ratio)
    try:
        yield
    finally:
        first_linear.weight.data.copy_(orig_weight)
