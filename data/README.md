# Benchmark data

Test splits used by the paper's pruning curves (copied unchanged from the evaluation harness the paper used,
which follows the Qwen2.5-Math / SimpleRL-Zoo evaluation data):

| directory  | file         | problems | source | used in |
|------------|--------------|---------:|--------|---------|
| `math500/` | `test.jsonl` |      500 | MATH-500 subset of MATH (Hendrycks et al., 2021; Lightman et al., 2023) | Figs. 2, 4, 8 |
| `gsm8k/`   | `test.jsonl` |     1319 | GSM8K test split (Cobbe et al., 2021) | Fig. 2 (and Fig. 5) |
| `arc/`     | `arc.json`   |     1172 | ARC-Challenge test split (Clark et al., 2018), verl-style records | Fig. 4 |

Each line is one JSON record; problems are identified by their line number (`idx`).
