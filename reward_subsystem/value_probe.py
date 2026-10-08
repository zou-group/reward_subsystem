"""Value probe: data, hidden-state extraction, TD training, question-only scoring and AUC.

The value probe reads one hidden state per token, ``h_t = hidden_states[layer][t]``
(HF convention: index 0 is the embedding output, index ``L`` the output of
decoder block ``L-1``), and predicts a value ``V(h_t)``. It is trained with a
token-level TD(0) loss on ``prompt + response`` texts whose terminal reward is
the binary correctness label ``R``::

    loss = mean over tokens t >= s of  (gamma * V_{t+1} - V_t)^2,   last token: (gamma * R - V_T)^2

where ``s`` is the first token of the user prompt (chat template) or 0.

At evaluation the probe scores the *question only*: ``V(s_0)`` is the value at
the last token of the prompt text (no response), compared against ``R`` by ROC-AUC.

The probe text always uses the ``cot`` prompt (``Question: ...\\nAnswer:``), as in
the paper, whatever template produced the step-1 response.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from reward_subsystem.benchmarks import build_prompts
from reward_subsystem.outputs import load_binary_rewards
from reward_subsystem.text_format import probe_text, user_start_token_index


VALUE_PROMPT_TYPE = "cot"


@dataclass
class LabeledProblem:
    idx: int
    prompt: str
    response: str
    reward: float


@dataclass
class HiddenStateBatch:
    """Hidden states of one padded batch, cached on CPU in the model dtype."""
    hidden: torch.Tensor          # (B, T, H)
    lengths: List[int]            # non-padding tokens per row
    starts: List[int]             # first scored token per row
    rewards: List[float]
    idx: List[int]


def load_labeled_problems(outputs_jsonl: str, benchmark: str, data_dir: str) -> List[LabeledProblem]:
    """Problems that have a model output, sorted by ``idx``, with string-equality rewards."""
    reward_map, response_map = load_binary_rewards(outputs_jsonl)
    samples = [s for s in build_prompts(benchmark, data_dir, VALUE_PROMPT_TYPE) if s["idx"] in reward_map]
    samples.sort(key=lambda s: s["idx"])
    return [
        LabeledProblem(idx=s["idx"], prompt=s["prompt"], response=response_map[s["idx"]],
                       reward=np.float32(reward_map[s["idx"]]))
        for s in samples
    ]


def train_valid_split(n: int, train_ratio: float = 0.8, seed: int = 0):
    """Positions of the train / validation problems (random permutation, then 80/20 cut)."""
    perm = np.random.RandomState(seed).permutation(n)
    n_train = int(n * train_ratio)
    return perm[:n_train], perm[n_train:]


def extract_hidden_states(model, tokenizer, problems: Sequence[LabeledProblem], positions,
                          layer: int, apply_chat_template: bool, with_response: bool,
                          batch_size: int, max_length: int, device) -> List[HiddenStateBatch]:
    """Run the frozen LLM once over ``problems[positions]`` (in that order, ``batch_size`` per batch)."""
    batches = []
    for start in range(0, len(positions), batch_size):
        batch = [problems[i] for i in positions[start:start + batch_size]]
        texts = [probe_text(tokenizer, p.prompt, p.response if with_response else None,
                            apply_chat_template) for p in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, use_cache=False, output_hidden_states=True)
        attn = enc["attention_mask"]

        lengths, starts = [], []
        for b, p in enumerate(batch):
            lengths.append(int(attn[b].sum().item()))
            starts.append(user_start_token_index(texts[b], p.prompt, attn[b].cpu(), offsets[b])
                          if apply_chat_template else 0)
        batches.append(HiddenStateBatch(
            hidden=out.hidden_states[layer].cpu(),
            lengths=lengths, starts=starts,
            rewards=[p.reward for p in batch], idx=[p.idx for p in batch],
        ))
        del out
    return batches


def td_loss(values: torch.Tensor, batch: HiddenStateBatch, rewards: torch.Tensor, gamma: float):
    """Sum of squared TD errors over the scored tokens of a batch, and the number of terms."""
    loss, n_terms = 0.0, 0
    for b in range(values.size(0)):
        length, start = batch.lengths[b], batch.starts[b]
        if length == 0 or start >= length:
            continue
        v = values[b, start:length]
        if v.numel() > 1:
            deltas_inner = gamma * v[1:] - v[:-1]
            loss += (deltas_inner ** 2).sum()
            n_terms += deltas_inner.numel()
        delta_eos = gamma * rewards[b] - v[-1]
        loss += delta_eos ** 2
        n_terms += 1
    return loss, n_terms


def run_td_epoch(probe, batches: Sequence[HiddenStateBatch], gamma: float, device,
                 optimizer: Optional[torch.optim.Optimizer] = None) -> float:
    """One pass over ``batches`` (an update per batch if ``optimizer`` is given). Returns mean TD loss."""
    train = optimizer is not None
    probe.train(train)
    total_loss, total_terms = 0.0, 0
    with torch.set_grad_enabled(train):
        for batch in batches:
            values = probe(batch.hidden.to(device).to(torch.float32))
            rewards = torch.tensor(batch.rewards, dtype=torch.float32, device=device)
            loss, n_terms = td_loss(values, batch, rewards, gamma)
            if n_terms == 0:
                continue
            loss = loss / n_terms
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * n_terms
            total_terms += n_terms
    return total_loss / max(total_terms, 1)


@torch.no_grad()
def score_last_token(probe, batches: Sequence[HiddenStateBatch], device):
    """Value at the last non-padding token of each row: ``(idx, reward, score)`` lists."""
    probe.eval()
    idx, rewards, scores = [], [], []
    for batch in batches:
        values = probe(batch.hidden.to(device).to(torch.float32))
        for b in range(values.size(0)):
            length, start = batch.lengths[b], batch.starts[b]
            if length == 0 or start >= length:
                continue
            idx.append(batch.idx[b])
            rewards.append(int(batch.rewards[b]))
            scores.append(float(values[b, start:length][-1].item()))
    return idx, rewards, scores


def auc_with_bootstrap_std(labels, scores, seed: int = 0, n_bootstrap: int = 1000):
    """ROC-AUC of ``scores`` against binary ``labels`` and the std of bootstrap AUCs.

    The bootstrap uses a fresh ``RandomState(seed)`` on every call, so all points
    of a curve share the same resamples. Resamples containing one class only are skipped.
    """
    auc = roc_auc_score(labels, scores)
    labels_arr, scores_arr = np.array(labels), np.array(scores)
    rng = np.random.RandomState(seed)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(labels_arr), size=len(labels_arr), replace=True)
        if len(np.unique(labels_arr[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(labels_arr[idx], scores_arr[idx]))
    std = float(np.std(boot_aucs)) if boot_aucs else 0.0
    return auc, std
