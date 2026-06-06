#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(/home/pah2704/miniconda3/envs/ai_env/bin/python)
fi

run_train() {
  local config="$1"
  local metrics_dir="$2"
  local fold="$3"
  local seed="$4"
  local tag="$5"
  local notes="$6"
  local summary="${metrics_dir}/${fold}_seed${seed}_summary.json"

  if [[ -f "${summary}" ]]; then
    echo "[skip] ${summary}"
    return 0
  fi

  echo "[run] config=${config} fold=${fold} seed=${seed}"
  "${PYTHON_CMD[@]}" -m src.train_multimodal \
    --config "${config}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --no-wandb \
    --journal-tag "${tag}" \
    --journal-notes "${notes}"
}

# 1) Single-stream ablation requested for Fold1 and Fold2.
for fold in fold1 fold2; do
  run_train \
    configs/paper_ablation_static.yaml \
    outputs/metrics/paper_ablation_static \
    "${fold}" \
    42 \
    paper_single_stream \
    "ICDSAIA ablation: static/spatial-only clean protocol, Fold1-Fold2 mean."

  run_train \
    configs/paper_ablation_flow.yaml \
    outputs/metrics/paper_ablation_flow \
    "${fold}" \
    42 \
    paper_single_stream \
    "ICDSAIA ablation: optical-flow-only clean protocol, Fold1-Fold2 mean."

  run_train \
    configs/paper_ablation_audio.yaml \
    outputs/metrics/paper_ablation_audio \
    "${fold}" \
    42 \
    paper_single_stream \
    "ICDSAIA ablation: audio-only clean protocol, Fold1-Fold2 mean."
done

# 2) Prior-KL contribution and random-seed robustness.
for seed in 42 123 2025; do
  for fold in fold1 fold2 fold3; do
    run_train \
      configs/paper_gated_prior_kl.yaml \
      outputs/metrics/paper_gated_prior_kl \
      "${fold}" \
      "${seed}" \
      paper_gated_prior_kl_multiseed \
      "ICDSAIA main model: gated logits with temporal face-valid mask and Prior-KL."

    run_train \
      configs/paper_gated_neutral_no_prior.yaml \
      outputs/metrics/paper_gated_neutral_no_prior \
      "${fold}" \
      "${seed}" \
      paper_gated_neutral_no_prior_multiseed \
      "ICDSAIA baseline: gated logits temporal-mask model with uniform gate initialization and no Prior-KL regularizer."

    run_train \
      configs/paper_gated_no_prior.yaml \
      outputs/metrics/paper_gated_no_prior \
      "${fold}" \
      "${seed}" \
      paper_gated_prior_init_no_kl_multiseed \
      "ICDSAIA baseline: identical gated logits temporal-mask model with prior gate initialization but without Prior-KL regularizer."
  done
done

"${PYTHON_CMD[@]}" scripts/summarize_paper_ablation.py
