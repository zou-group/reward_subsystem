"""Model-output files (written by ``scripts/01_generate_responses.py``) and the labels derived from them.

One JSON line per problem::

    {"idx": 0, "question": ..., "gt_cot": ..., "gt": "...",   # + benchmark extras (level, type, ...)
     "code":  [response],                                      # one sampled response
     "pred":  [extracted_answer],
     "report": [null],
     "score": [true/false]}                                    # math_equal(pred, gt)

The probe label is NOT the ``score`` field: as in the paper's code, it is the
**exact string equality** ``str(pred[0]) == str(gt)`` (``load_binary_rewards``).
"""

import json
from typing import Dict, Tuple

from reward_subsystem.benchmarks import read_jsonl


def output_filename(prompt_type: str, max_samples: int, seed: int, temperature: float) -> str:
    """File name used by the generation step, e.g. ``test_qwen-boxed_-1_seed0_t1.0_s0_e-1.jsonl``."""
    return f"test_{prompt_type}_{max_samples}_seed{seed}_t{temperature}_s0_e-1.jsonl"


def write_jsonl(records, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_binary_rewards(outputs_jsonl: str) -> Tuple[Dict[int, int], Dict[int, str]]:
    """Return ``(reward_map, response_map)`` keyed by problem ``idx``.

    reward = 1 iff the first extracted answer equals the reference answer as a string;
    response = the first sampled response.
    """
    reward_map, response_map = {}, {}
    for rec in read_jsonl(outputs_jsonl):
        idx = int(rec["idx"])
        preds = rec.get("pred", [])
        pred0 = str(preds[0]) if len(preds) > 0 else ""
        reward_map[idx] = 1 if pred0 == str(rec["gt"]) else 0
        response_map[idx] = rec.get("code", [""])[0]
    return reward_map, response_map
