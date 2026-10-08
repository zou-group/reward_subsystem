"""Dopamine neurons: TD errors at paragraph boundaries, paragraph features, and the dopamine probe.

Pipeline (one model, one benchmark):

* **Step 3 - TD errors** (``scripts/03_dopamine_td_errors.py``). Each response is split into
  paragraphs ``p_1..p_n``. For every boundary ``t = 0..n-1`` we sample
  ``n_rollouts`` continuations from ``prompt + p_1..p_t`` and set
  ``V(t)`` = fraction of continuations whose answer is ``math_equal`` to the
  reference; ``V(n)`` = the response's own (string-equality) reward. The TD
  error of paragraph ``t+1`` is ``delta(t) = gamma * V(t+1) - V(t)``.
* **Step 4 - paragraph features** (``scripts/04_dopamine_features.py``). One forward pass over
  ``prompt + response``; hidden states of *all* decoder blocks (HF
  ``hidden_states[1:]``) over the response tokens are z-scored per response
  and averaged over each paragraph's tokens.
* **Step 5 - dopamine probe** (``scripts/05_train_dopamine_probe.py``). For one layer, keep the
  paragraphs with ``|delta| > td_threshold``, split by problem, and regress
  ``delta`` with a small MLP. Its strongest input columns are the dopamine neurons.
* **Step 7 - Spearman vs pruning** (``scripts/07_plot_dopamine_spearman_vs_prune.py``) reads the
  held-out rows again with ``load_dopamine_dataset`` and ``rows_of_problems``.

Prompt: steps 3-4 rebuild every prompt with the ``qwen-boxed`` template, as in the
paper (``DOPAMINE_PROMPT_TYPE``), whatever template produced the step-1 response.

Layer convention: dopamine ``layer = L`` is the output of decoder block ``L``
(0-based), i.e. HF ``hidden_states[L + 1]``. This differs by one from the value
probe's ``layer`` argument, which indexes ``hidden_states`` directly.
"""

import multiprocessing
import multiprocessing.pool
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr

from reward_subsystem.answer_parsing import extract_answer
from reward_subsystem.benchmarks import build_prompts
from reward_subsystem.grading import math_equal
from reward_subsystem.outputs import load_binary_rewards
from reward_subsystem.probes import MLPProbe
from reward_subsystem.text_format import chat_text, rollout_prefix, split_into_paragraphs


CACHE_RELPATH = os.path.join(".cache", "hidden_states_all_layers.npz")
DOPAMINE_PROMPT_TYPE = "qwen-boxed"


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------

@dataclass
class RolloutProblem:
    idx: int
    prompt: str
    gt: str
    reward: int
    paragraphs: List[str]


def load_rollout_problems(outputs_jsonl: str, benchmark: str, data_dir: str,
                          max_samples: int = -1) -> List[RolloutProblem]:
    """Problems with a non-empty response, sorted by ``idx`` (the order of every later array)."""
    reward_map, response_map = load_binary_rewards(outputs_jsonl)
    samples = {s["idx"]: s for s in build_prompts(benchmark, data_dir, DOPAMINE_PROMPT_TYPE)
               if s["idx"] in reward_map}
    indices = sorted(samples)
    if max_samples > 0:
        indices = indices[:max_samples]

    problems = []
    for idx in indices:
        response = response_map.get(idx, "")
        if not response.strip():
            continue
        paragraphs = split_into_paragraphs(response)
        if len(paragraphs) == 0:
            continue
        problems.append(RolloutProblem(idx=idx, prompt=samples[idx]["prompt"], gt=samples[idx]["gt"],
                                       reward=reward_map[idx], paragraphs=paragraphs))
    return problems


# ---------------------------------------------------------------------------
# Step 3 - Monte-Carlo values and TD errors
# ---------------------------------------------------------------------------

def boundary_prefixes(tokenizer, problem: RolloutProblem) -> List[str]:
    """Generation prefixes for boundaries ``t = 0..n-1`` (``t`` paragraphs already written)."""
    return [
        rollout_prefix(tokenizer, problem.prompt, "\n\n".join(problem.paragraphs[:t]) if t > 0 else "")
        for t in range(len(problem.paragraphs))
    ]


class _NoDaemonProcess(multiprocessing.Process):
    """Pool worker that may start child processes (``math_equal`` uses a timeout subprocess)."""

    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonPool(multiprocessing.pool.Pool):
    @staticmethod
    def Process(ctx, *args, **kwds):
        return _NoDaemonProcess(*args, **kwds)


def _grade_one(task):
    text, gt = task
    return math_equal(extract_answer(text), gt, timeout=True)


def td_errors_from_rollouts(problem: RolloutProblem, completions: Dict[str, List[str]],
                            gamma: float, pool) -> dict:
    """Grade the rollouts of one problem and compute ``V(t)`` and ``delta(t)``.

    ``completions`` maps ``str(t)`` to the rollout texts from boundary ``t``.
    """
    n = len(problem.paragraphs)
    values = [None] * (n + 1)
    values[n] = float(problem.reward)

    tasks, counts = [], []
    for t_str, texts in completions.items():
        tasks.extend((text, problem.gt) for text in texts)
        counts.append((int(t_str), len(texts)))
    correct = pool.map(_grade_one, tasks, chunksize=64)

    pos = 0
    for t, n_texts in counts:
        values[t] = sum(correct[pos:pos + n_texts]) / n_texts
        pos += n_texts
    assert all(v is not None for v in values), f"Missing V(t) for sample {problem.idx}"

    values_arr = np.array(values, dtype=np.float64)
    td_errors = gamma * values_arr[1:] - values_arr[:-1]
    return {
        "idx": problem.idx,
        "reward": problem.reward,
        "sentences": problem.paragraphs,
        "values": values,
        "td_errors": td_errors.tolist(),
    }


# ---------------------------------------------------------------------------
# Step 4 - paragraph features from all layers
# ---------------------------------------------------------------------------

@torch.no_grad()
def paragraph_features(model, tokenizer, prompt: str, paragraphs: List[str], max_length: int = 4096):
    """Per-paragraph features of one response, shape ``(n_paragraphs, n_layers * hidden_dim)``.

    * Hidden states of all decoder blocks (``hidden_states[1:]``) are concatenated
      for the response tokens ``[prompt_len, total_len)``.
    * ``prompt_len`` is the token length of the prompt followed by an *empty, closed*
      assistant turn, so for Qwen the window starts 2 tokens into the response
      (paper behaviour, kept as is).
    * Each feature is z-scored over the response tokens, then averaged over the
      tokens of each paragraph (paragraph ``t`` ends where the text of the first
      ``t`` paragraphs ends).
    """
    device = next(model.parameters()).device
    full_text = chat_text(tokenizer, prompt, "\n\n".join(paragraphs))
    full_enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    full_enc = {k: v.to(device) for k, v in full_enc.items()}
    prompt_len = tokenizer(chat_text(tokenizer, prompt, ""), return_tensors="pt")["input_ids"].shape[1]

    out = model(**full_enc, use_cache=False, output_hidden_states=True)
    n_layers = len(out.hidden_states) - 1
    hidden_dim = out.hidden_states[1].shape[-1]

    total_len = int(full_enc["attention_mask"][0].sum().item())
    response_start = min(prompt_len, total_len - 1)
    hidden = torch.cat(
        [hs[0, response_start:total_len] for hs in out.hidden_states[1:]], dim=-1
    ).to(torch.float32).cpu().numpy()
    del out

    mean = hidden.mean(axis=0, keepdims=True)
    std = hidden.std(axis=0, keepdims=True) + 1e-8
    normed = (hidden - mean) / std

    features = []
    prev_end = response_start
    for t in range(1, len(paragraphs) + 1):
        text_t = chat_text(tokenizer, prompt, "\n\n".join(paragraphs[:t]))
        cur_end = int(tokenizer(text_t, return_tensors="pt", truncation=True,
                                max_length=max_length)["input_ids"].shape[1])
        start_tok, end_tok = prev_end - response_start, cur_end - response_start
        if start_tok >= end_tok:
            start_tok = max(0, end_tok - 1)
        features.append(normed[start_tok:end_tok].mean(axis=0))
        prev_end = cur_end
    return features, n_layers, hidden_dim


# ---------------------------------------------------------------------------
# Steps 5 and 7 - dopamine probe data and training
# ---------------------------------------------------------------------------

def load_dopamine_dataset(cache_path: str, layer: int, td_threshold: float):
    """Features of one layer and TD errors of the paragraphs with ``|delta| > td_threshold``.

    Returns ``(td_errors, features, sample_idxs, hidden_dim)``.
    """
    cache = np.load(cache_path)
    n_layers, hidden_dim = int(cache["n_layers"]), int(cache["hidden_dim"])
    assert 0 <= layer < n_layers, f"layer {layer} out of range (model has {n_layers} layers)"
    td_errors = cache["td_errors"]
    features = cache["hidden_states"][:, layer * hidden_dim:(layer + 1) * hidden_dim]
    sample_idxs = cache["sample_idxs"]
    keep = np.abs(td_errors) > td_threshold
    return td_errors[keep], features[keep], sample_idxs[keep], hidden_dim


def problem_level_split(sample_idxs: np.ndarray, train_ratio: float = 0.8, seed: int = 0):
    """Split paragraphs by problem: shuffle the problem ids, first ``train_ratio`` go to train.

    Returns ``(train_rows, test_rows, test_problems)`` (row indices in ascending order).
    """
    problems = np.unique(sample_idxs)
    np.random.RandomState(seed).shuffle(problems)
    n_train = int(len(problems) * train_ratio)
    train_problems = set(problems[:n_train].tolist())
    test_problems = set(problems[n_train:].tolist())
    train_rows = rows_of_problems(sample_idxs, train_problems)
    test_rows = rows_of_problems(sample_idxs, test_problems)
    return train_rows, test_rows, sorted(test_problems)


def rows_of_problems(sample_idxs: np.ndarray, problem_ids) -> np.ndarray:
    """Row indices (ascending) of the paragraphs that belong to ``problem_ids``."""
    return np.where(np.isin(sample_idxs, list(problem_ids)))[0]


def train_dopamine_probe(X_train, y_train, X_test, y_test, mlp_hidden: int, num_epochs: int,
                         batch_size: int, lr: float, weight_decay: float, device):
    """Regress TD errors with an MLP probe (MSE, AdamW, shuffled mini-batches).

    The test MSE is checked at epoch 1, every 10th epoch and the last epoch; the
    probe with the lowest test MSE is restored and returned (model selection on
    the test split, as in the paper's code).

    Returns ``(probe, best_state_dict_on_cpu)``.
    """
    probe = MLPProbe(X_train.shape[1], mlp_hidden).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    n_batches = (len(X_train_t) + batch_size - 1) // batch_size
    best_test_loss, best_state = float("inf"), None
    for ep in range(1, num_epochs + 1):
        probe.train()
        perm = torch.randperm(len(X_train_t), device=device)
        epoch_loss = 0.0
        for bi in range(n_batches):
            rows = perm[bi * batch_size:(bi + 1) * batch_size]
            loss = criterion(probe(X_train_t[rows]), y_train_t[rows])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches

        if ep % 10 == 0 or ep == 1 or ep == num_epochs:
            probe.eval()
            with torch.no_grad():
                pred_test = probe(X_test_t).cpu().numpy()
                test_loss = criterion(torch.tensor(pred_test), y_test_t.cpu()).item()
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            print(f"  epoch {ep:4d}: train_mse={epoch_loss:.6f} test_mse={test_loss:.6f} "
                  f"test_pearson={pearsonr(y_test, pred_test)[0]:.4f} "
                  f"test_spearman={spearmanr(y_test, pred_test)[0]:.4f}", flush=True)

    if best_state is None:
        raise RuntimeError("No probe was selected: num_epochs < 1 or the test MSE was NaN at every check.")
    probe.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    probe.eval()
    print(f"  restored best probe (test_mse={best_test_loss:.6f})")
    return probe, best_state
