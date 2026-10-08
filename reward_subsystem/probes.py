"""The probe architecture and its checkpoints.

Both probes in the paper are the same two-layer MLP

    input_dim -> hidden_dim -> ReLU -> 1

* value probe   : input = one hidden state (per token), hidden_dim 1024,
                  trained with a TD loss (``value_probe.py``).
* dopamine probe: input = paragraph-averaged hidden state of one layer,
                  hidden_dim 32, trained with MSE on TD errors (``dopamine.py``).

Neurons are ranked by the weights of the first linear layer (``pruning.py``).
Parameter names (``mlp.0.weight`` ...) match the paper's checkpoints, so those
checkpoints load directly.
"""

import torch
import torch.nn as nn


class MLPProbe(nn.Module):
    """Two-layer MLP mapping ``(..., input_dim)`` to a scalar per position ``(...)``."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @property
    def first_linear(self) -> nn.Linear:
        """The layer whose input columns define value / dopamine neurons."""
        return self.mlp[0]

    def forward(self, h):
        return self.mlp(h).squeeze(-1)


def save_value_probe(path, probe: MLPProbe, hidden_size: int, mlp_hidden: int,
                     gamma: float, epoch: int, val_loss: float, split: dict) -> None:
    """Save a value probe.

    ``split`` records the train / validation split of step 2
    (``{"train_ratio", "seed", "valid_idx"}``, ``valid_idx`` in split order) so
    that step 6 scores exactly the validation problems the probe was selected on.
    """
    torch.save({
        "model_state_dict": probe.state_dict(),
        "hidden_size": hidden_size,
        "mlp_hidden": mlp_hidden,
        "gamma": gamma,
        "epoch": epoch,
        "val_loss": val_loss,
        "split": split,
    }, path)


def load_value_probe(path, device):
    """Returns ``(probe, checkpoint_dict)``. Checkpoints of the paper's code have no ``split`` entry."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    probe = MLPProbe(ckpt["hidden_size"], ckpt.get("mlp_hidden", 1024))
    probe.load_state_dict(ckpt["model_state_dict"])
    return probe.to(device).eval(), ckpt


def load_dopamine_probe(path, device):
    """Returns ``(probe, checkpoint_dict)``."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    probe = MLPProbe(ckpt["input_dim"], ckpt["mlp_hidden"])
    probe.load_state_dict(ckpt["model_state_dict"])
    return probe.to(device).eval(), ckpt
