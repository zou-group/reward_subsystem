#!/usr/bin/env bash
# End-to-end pipeline for one model on one benchmark:
#   1 responses -> 2 value probes -> 3-5 dopamine probes -> 6 value AUC figure -> 7 dopamine Spearman figure
# Run from the repository root:  bash examples/run_all.sh
set -euo pipefail

# ---------------------------------------------------------------- settings
ARTIFACT_DIR=${ARTIFACT_DIR:-./artifacts}           # everything generated goes here
MODEL=${MODEL:-hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo}
TAG=${TAG:-7B_simplerl}                             # short name used in directory names
BENCHMARK=${BENCHMARK:-math500}                     # math500, gsm8k or arc
PROMPT_TYPE=${PROMPT_TYPE:-qwen-boxed}              # llama-boxed for Llama / Phi
GEN_FLAGS=${GEN_FLAGS:-}                            # "--apply_chat_template" for Llama / Phi
VALUE_FLAGS=${VALUE_FLAGS:-}                        # "--apply_chat_template" to probe chat-formatted text
VALUE_LAYERS=${VALUE_LAYERS:-"2 3 4"}               # HF hidden_states index
DOPAMINE_LAYERS=${DOPAMINE_LAYERS:-"20 21 22"}      # 0-based decoder block

OUTPUTS=$ARTIFACT_DIR/outputs/output_${TAG}_16000tk/$BENCHMARK/test_${PROMPT_TYPE}_-1_seed0_t1.0_s0_e-1.jsonl
TD_DIR=$ARTIFACT_DIR/dopamine/td_dopamine_${TAG}_${BENCHMARK}
FIG_DIR=$ARTIFACT_DIR/figures
# The paper ran the dopamine pipeline (steps 3-5, 7) only for the Qwen2.5 RL models on MATH-500
# with qwen-boxed outputs; it is skipped otherwise.
RUN_DOPAMINE=0; [ "$PROMPT_TYPE" = qwen-boxed ] && RUN_DOPAMINE=1

# ---------------------------------------------------------------- 1. responses (vLLM)
python scripts/01_generate_responses.py --model "$MODEL" --benchmark "$BENCHMARK" \
    --prompt_type "$PROMPT_TYPE" $GEN_FLAGS --output_dir "$ARTIFACT_DIR/outputs/output_${TAG}_16000tk"

# ---------------------------------------------------------------- 2. value probes
for L in $VALUE_LAYERS; do
    python scripts/02_train_value_probe.py --model "$MODEL" --outputs_jsonl "$OUTPUTS" \
        --benchmark "$BENCHMARK" --layer "$L" $VALUE_FLAGS \
        --output_dir "$ARTIFACT_DIR/value_probes/${TAG}_${BENCHMARK}/layer_$L"
done

# ---------------------------------------------------------------- 3-5. dopamine probes
if [ "$RUN_DOPAMINE" = 1 ]; then
    python scripts/03_dopamine_td_errors.py --model "$MODEL" --outputs_jsonl "$OUTPUTS" \
        --benchmark "$BENCHMARK" --result_dir "$TD_DIR"
    python scripts/04_dopamine_features.py --model "$MODEL" --outputs_jsonl "$OUTPUTS" \
        --benchmark "$BENCHMARK" --result_dir "$TD_DIR"
    for L in $DOPAMINE_LAYERS; do
        python scripts/05_train_dopamine_probe.py --result_dir "$TD_DIR" --layer "$L" \
            --output_dir "$ARTIFACT_DIR/dopamine_probes/${TAG}_${BENCHMARK}/layer$L"
    done
fi

# ---------------------------------------------------------------- 6. value probe: AUC vs pruning ratio
AUC_CURVES=()
for L in $VALUE_LAYERS; do
    C=$ARTIFACT_DIR/value_auc/${TAG}_${BENCHMARK}_layer$L.json
    python scripts/06_plot_value_auc_vs_prune.py curve --model "$MODEL" --outputs_jsonl "$OUTPUTS" \
        --benchmark "$BENCHMARK" --layer "$L" $VALUE_FLAGS \
        --probe "$ARTIFACT_DIR/value_probes/${TAG}_${BENCHMARK}/layer_$L/value_probe_best.pt" \
        --output_json "$C"
    AUC_CURVES+=("Layer $L=$C")
done
python scripts/06_plot_value_auc_vs_prune.py plot --output "$FIG_DIR/${TAG}_${BENCHMARK}_value_auc.pdf" \
    --panel "$BENCHMARK" "${AUC_CURVES[@]}"

# ---------------------------------------------------------------- 7. dopamine probe: Spearman vs pruning ratio
if [ "$RUN_DOPAMINE" = 1 ]; then
    RHO_CURVES=()
    for L in $DOPAMINE_LAYERS; do
        D=$ARTIFACT_DIR/dopamine_probes/${TAG}_${BENCHMARK}/layer$L
        python scripts/07_plot_dopamine_spearman_vs_prune.py curve --result_dir "$TD_DIR" --probe_dir "$D" \
            --output_json "$D/spearman_curve.json"
        RHO_CURVES+=("Layer $L=$D/spearman_curve.json")
    done
    python scripts/07_plot_dopamine_spearman_vs_prune.py plot \
        --output "$FIG_DIR/${TAG}_${BENCHMARK}_dopamine_spearman.pdf" --panel "$BENCHMARK" "${RHO_CURVES[@]}"
fi
