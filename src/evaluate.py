from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import append_experiment_rows, ensure_dir, log_args, render_experiment_journal, setup_logger, write_json


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_lie": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_lie": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_lie": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
        "threshold": float(threshold),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_score))
        metrics["auc_pr"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auc_roc"] = float("nan")
        metrics["auc_pr"] = float("nan")
    return metrics


def threshold_metric_value(y_true: np.ndarray, y_score: np.ndarray, threshold: float, metric: str) -> float:
    y_pred = (y_score >= threshold).astype(int)
    normalized = metric.lower().removeprefix("val_")
    aliases = {
        "balanced_acc": "balanced_accuracy",
        "bal_acc": "balanced_accuracy",
        "f1": "f1_lie",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if normalized == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if normalized == "f1_lie":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if normalized == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if normalized == "precision_lie":
        return float(precision_score(y_true, y_pred, zero_division=0))
    if normalized == "recall_lie":
        return float(recall_score(y_true, y_pred, zero_division=0))
    raise ValueError(f"Unsupported threshold calibration metric: {metric}")


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "balanced_accuracy") -> dict[str, float | str]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    unique_scores = np.unique(y_score)
    if unique_scores.size == 0:
        return {"threshold": 0.5, "metric": metric, "metric_value": 0.0}

    eps = 1e-12
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0 if unique_scores.size > 1 else np.array([])
    candidates = np.unique(
        np.concatenate(
            [
                np.array([0.5, unique_scores[0] - eps, unique_scores[-1] + eps]),
                unique_scores,
                midpoints,
            ]
        )
    )

    best_threshold = 0.5
    best_value = -1.0
    best_tiebreak = -1.0
    for threshold in candidates:
        value = threshold_metric_value(y_true, y_score, float(threshold), metric)
        # Prefer less class-collapsed thresholds when the primary metric ties.
        tiebreak = threshold_metric_value(y_true, y_score, float(threshold), "macro_f1")
        if value > best_value or (np.isclose(value, best_value) and tiebreak > best_tiebreak):
            best_value = value
            best_tiebreak = tiebreak
            best_threshold = float(threshold)
    return {"threshold": best_threshold, "metric": metric, "metric_value": float(best_value)}


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"accuracy": [], "f1_lie": [], "auc_roc": []}
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        metrics = binary_metrics(y_true[idx], y_score[idx], threshold=threshold)
        for key in values:
            values[key].append(float(metrics[key]))
    return {
        key: {
            "low": float(np.percentile(vals, 2.5)),
            "high": float(np.percentile(vals, 97.5)),
        }
        for key, vals in values.items()
        if vals
    }


def topk_mean(scores: pd.Series, k: int = 3) -> float:
    values = np.sort(scores.astype(float).to_numpy())
    return float(values[-min(k, len(values)) :].mean())


def aggregate_windows(
    df: pd.DataFrame,
    video_col: str = "video_id",
    label_col: str = "label",
    score_col: str = "score_lie",
) -> dict[str, pd.DataFrame]:
    grouped = df.groupby(video_col, sort=True)
    base = grouped[label_col].first().rename("label").to_frame()
    outputs: dict[str, pd.DataFrame] = {}
    for name, series in {
        "mean": grouped[score_col].mean(),
        "max": grouped[score_col].max(),
        "top3_mean": grouped[score_col].apply(topk_mean),
    }.items():
        out = base.copy()
        out["score_lie"] = series.astype(float)
        out = out.reset_index()
        out["aggregation"] = name
        outputs[name] = out
    return outputs


def evaluate_dataframe(df: pd.DataFrame, label_col: str, score_col: str, bootstrap: bool, seed: int) -> dict[str, Any]:
    y_true = df[label_col].astype(int).to_numpy()
    y_score = df[score_col].astype(float).to_numpy()
    metrics = binary_metrics(y_true, y_score)
    if bootstrap:
        metrics["bootstrap_95ci"] = bootstrap_ci(y_true, y_score, seed=seed)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clip-level or window-level binary predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--out", default="outputs/metrics/evaluation.json")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--score-column", default="score_lie")
    parser.add_argument("--video-column", default="video_id")
    parser.add_argument("--window-level", action="store_true", help="Aggregate multiple rows per video.")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--journal-root", default="outputs/report_journal")
    parser.add_argument("--journal-tag", default="")
    parser.add_argument("--journal-notes", default="")
    parser.add_argument("--experiment-name", default="")
    return parser.parse_args()


def append_eval_journal(
    args: argparse.Namespace,
    metrics: dict[str, Any],
    out_path: Path,
) -> tuple[Path, Path]:
    journal_root = ensure_dir(args.journal_root)
    csv_path = journal_root / "experiment_journal.csv"
    md_path = journal_root / "experiment_journal.md"
    experiment_name = args.experiment_name or Path(args.predictions).stem
    rows: list[dict[str, Any]] = []
    if args.window_level:
        for aggregation, values in metrics.items():
            rows.append(
                {
                    "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "stage": "evaluate",
                    "experiment_name": experiment_name,
                    "tag": args.journal_tag,
                    "dataset": "dolos" if "dolos" in args.predictions.lower() else "",
                    "fold": "",
                    "seed": args.seed,
                    "aggregation": aggregation,
                    "accuracy": values.get("accuracy"),
                    "balanced_accuracy": values.get("balanced_accuracy"),
                    "precision_lie": values.get("precision_lie"),
                    "recall_lie": values.get("recall_lie"),
                    "f1_lie": values.get("f1_lie"),
                    "macro_f1": values.get("macro_f1"),
                    "auc_roc": values.get("auc_roc"),
                    "auc_pr": values.get("auc_pr"),
                    "metrics_json": str(out_path),
                    "predictions_path": args.predictions,
                    "notes": args.journal_notes,
                }
            )
    else:
        rows.append(
            {
                "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "stage": "evaluate",
                "experiment_name": experiment_name,
                "tag": args.journal_tag,
                "dataset": "dolos" if "dolos" in args.predictions.lower() else "",
                "fold": "",
                "seed": args.seed,
                "aggregation": "direct",
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "precision_lie": metrics.get("precision_lie"),
                "recall_lie": metrics.get("recall_lie"),
                "f1_lie": metrics.get("f1_lie"),
                "macro_f1": metrics.get("macro_f1"),
                "auc_roc": metrics.get("auc_roc"),
                "auc_pr": metrics.get("auc_pr"),
                "metrics_json": str(out_path),
                "predictions_path": args.predictions,
                "notes": args.journal_notes,
            }
        )
    append_experiment_rows(csv_path, rows)
    render_experiment_journal(csv_path, md_path)
    return csv_path, md_path


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("evaluate", args.log_dir)
    logger.info("Log file: %s", log_path)
    log_args(logger, args)
    df = pd.read_csv(args.predictions)
    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    if args.window_level:
        aggregated = aggregate_windows(df, args.video_column, args.label_column, args.score_column)
        metrics = {
            name: evaluate_dataframe(preds, "label", "score_lie", args.bootstrap, args.seed)
            for name, preds in aggregated.items()
        }
        for name, preds in aggregated.items():
            preds.to_csv(out_path.with_name(f"{out_path.stem}_{name}_predictions.csv"), index=False)
            logger.info("Wrote aggregated predictions: %s", out_path.with_name(f"{out_path.stem}_{name}_predictions.csv"))
    else:
        metrics = evaluate_dataframe(df, args.label_column, args.score_column, args.bootstrap, args.seed)
    write_json(out_path, metrics)
    logger.info("Wrote evaluation metrics to %s", out_path)
    logger.info("Metrics: %s", metrics)
    journal_csv, journal_md = append_eval_journal(args, metrics, out_path)
    logger.info("Experiment journal updated: %s", journal_csv)
    logger.info("Experiment journal markdown updated: %s", journal_md)


if __name__ == "__main__":
    main()
