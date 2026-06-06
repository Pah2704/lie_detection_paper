#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CMD=("${PYTHON_BIN}")
else
  PYTHON_CMD=(/home/pah2704/miniconda3/envs/ai_env/bin/python)
fi

CONFIG="${CONFIG:-configs/paper_gated_reliability_confidence.yaml}"
METRICS_DIR="${METRICS_DIR:-outputs/metrics/paper_gated_reliability_confidence}"
FOLDS="${FOLDS:-fold1 fold2 fold3}"
SEEDS="${SEEDS:-42}"
JOURNAL_TAG="${JOURNAL_TAG:-paper_gated_reliability_confidence}"
JOURNAL_NOTES="${JOURNAL_NOTES:-ICDSAIA reliability-confidence voting: modality weight equals reliability times stream confidence.}"

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
    --journal-tag "${JOURNAL_TAG}" \
    --journal-notes "${JOURNAL_NOTES}"
}

for seed in ${SEEDS}; do
  for fold in ${FOLDS}; do
    run_train "${fold}" "${seed}"
  done
done
