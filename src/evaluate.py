from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import ensure_dir, write_json


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | list[list[int]]]:
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_lie": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_lie": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_lie": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
        metrics["auc_pr"] = float(average_precision_score(y_true, y_score))
        fpr, tpr, _ = roc_curve(y_true, y_score)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fnr - fpr)))
        metrics["eer"] = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")
        metrics["eer"] = float("nan")
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auc_roc": [], "macro_f1": [], "recall_lie": []}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = binary_metrics(y_true[idx], y_score[idx])
        for key in values:
            values[key].append(float(m[key]))
    return {
        key: {
            "low": float(np.percentile(vals, 2.5)),
            "high": float(np.percentile(vals, 97.5)),
        }
        for key, vals in values.items()
        if vals
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate binary video-level predictions.")
    parser.add_argument("--predictions", required=True, help="CSV with label and score columns.")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--score-column", default="score_lie")
    parser.add_argument("--out", default="outputs/metrics/evaluation.json")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.predictions)
    y_true = df[args.label_column].astype(int).to_numpy()
    y_score = df[args.score_column].astype(float).to_numpy()
    metrics = binary_metrics(y_true, y_score)
    if args.bootstrap:
        metrics["bootstrap_95ci"] = bootstrap_ci(y_true, y_score, seed=args.seed)
    out = Path(args.out)
    ensure_dir(out.parent)
    write_json(out, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
