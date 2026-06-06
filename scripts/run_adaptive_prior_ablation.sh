#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(/home/pah2704/miniconda3/envs/ai_env/bin/python)
fi

CONFIG="${CONFIG:-configs/paper_gated_adaptive_prior_kl.yaml}"
METRICS_DIR="${METRICS_DIR:-outputs/metrics/paper_gated_adaptive_prior_kl}"
FOLDS="${FOLDS:-fold1 fold2 fold3}"
SEEDS="${SEEDS:-42}"

run_train() {
  local fold="$1"
  local seed="$2"
  local summary="${METRICS_DIR}/${fold}_seed${seed}_summary.json"

  if [[ -f "${summary}" ]]; then
    echo "[skip] ${summary}"
    return 0
  fi

  echo "[run] config=${CONFIG} fold=${fold} seed=${seed}"
  "${PYTHON_CMD[@]}" -m src.train_multimodal \
    --config "${CONFIG}" \
    --fold "${fold}" \
    --seed "${seed}" \
    --no-wandb \
    --journal-tag paper_gated_adaptive_prior_kl \
    --journal-notes "ICDSAIA adaptive prior: face-valid-conditioned Prior-KL for gated logit fusion."
}

for seed in ${SEEDS}; do
  for fold in ${FOLDS}; do
    run_train "${fold}" "${seed}"
  done
done
