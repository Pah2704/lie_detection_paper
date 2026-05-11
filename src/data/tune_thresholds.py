from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import binary_metrics, bootstrap_ci
from src.utils import ensure_dir, write_json


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def average_predictions(paths: list[str | Path]) -> pd.DataFrame:
    dfs = [pd.read_csv(path) for path in paths]
    base = dfs[0][["video_id", "group_id", "label"]].copy()
    scores = np.stack([df["score_lie"].to_numpy(dtype=float) for df in dfs], axis=0)
    base["score_lie"] = scores.mean(axis=0)
    return base


def fuse_predictions(visual_df: pd.DataFrame, text_df: pd.DataFrame, alpha_visual: float) -> pd.DataFrame:
    merged = visual_df.merge(
        text_df[["video_id", "score_lie"]].rename(columns={"score_lie": "text_score_lie"}),
        on="video_id",
        how="inner",
    )
    merged["score_lie"] = alpha_visual * merged["score_lie"] + (1.0 - alpha_visual) * merged["text_score_lie"]
    return merged[["video_id", "group_id", "label", "score_lie"]]


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    mids = (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else unique
    candidates = np.concatenate(([0.5], unique, mids))
    return np.unique(candidates)


def youden_j(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> float:
    y_pred = (y_score >= threshold).astype(int)
    positives = y_true == 1
    negatives = y_true == 0
    tpr = float((y_pred[positives] == 1).mean()) if positives.any() else 0.0
    fpr = float((y_pred[negatives] == 1).mean()) if negatives.any() else 0.0
    return tpr - fpr


def select_threshold(val_df: pd.DataFrame, optimize_metric: str) -> dict[str, Any]:
    y_true = val_df["label"].to_numpy(dtype=int)
    y_score = val_df["score_lie"].to_numpy(dtype=float)
    best: dict[str, Any] | None = None
    for threshold in candidate_thresholds(y_score):
        metrics = binary_metrics(y_true, y_score, threshold=float(threshold))
        if optimize_metric == "youden_j":
            score = youden_j(y_true, y_score, float(threshold))
        else:
            score = float(metrics[optimize_metric])
        row = {"threshold": float(threshold), "selection_score": score, "metrics": metrics}
        if best is None:
            best = row
            continue
        better = score > float(best["selection_score"])
        tied_and_closer_to_half = score == float(best["selection_score"]) and abs(float(threshold) - 0.5) < abs(float(best["threshold"]) - 0.5)
        if better or tied_and_closer_to_half:
            best = row
    if best is None:
        raise ValueError("No threshold candidates were generated")
    return best


def load_default_models(args: argparse.Namespace) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    visual_val = average_predictions(
        [
            "outputs/metrics/baseline_frame_resnet18/val_predictions.csv",
            "outputs/metrics/baseline_frame_resnet18_seed123/val_predictions.csv",
            "outputs/metrics/baseline_frame_resnet18_seed2025/val_predictions.csv",
        ]
    )
    visual_test = average_predictions(
        [
            "outputs/metrics/baseline_frame_resnet18/test_predictions.csv",
            "outputs/metrics/baseline_frame_resnet18_seed123/test_predictions.csv",
            "outputs/metrics/baseline_frame_resnet18_seed2025/test_predictions.csv",
        ]
    )
    text_lr_val = pd.read_csv("outputs/metrics/real_life_text_baselines/tfidf_logistic_regression_val_predictions.csv")
    text_lr_test = pd.read_csv("outputs/metrics/real_life_text_baselines/tfidf_logistic_regression_test_predictions.csv")
    fusion_summary = read_json("outputs/metrics/real_life_late_fusion/summary.json")
    alpha_visual = float(fusion_summary["best_alpha_visual"])

    return {
        "resnet18_frame_ensemble": (visual_val, visual_test),
        "resnet18_seed42": (
            pd.read_csv("outputs/metrics/baseline_frame_resnet18/val_predictions.csv"),
            pd.read_csv("outputs/metrics/baseline_frame_resnet18/test_predictions.csv"),
        ),
        "r3d18_rgb": (
            pd.read_csv("outputs/metrics/rgb_r3d18/val_predictions.csv"),
            pd.read_csv("outputs/metrics/rgb_r3d18/test_predictions.csv"),
        ),
        "tfidf_logistic_regression": (text_lr_val, text_lr_test),
        "tfidf_linear_svm": (
            pd.read_csv("outputs/metrics/real_life_text_baselines/tfidf_linear_svm_val_predictions.csv"),
            pd.read_csv("outputs/metrics/real_life_text_baselines/tfidf_linear_svm_test_predictions.csv"),
        ),
        "late_fusion": (
            fuse_predictions(visual_val, text_lr_val, alpha_visual),
            pd.read_csv("outputs/metrics/real_life_late_fusion/test_predictions.csv")[["video_id", "group_id", "label", "score_lie"]],
        ),
    }


def run(args: argparse.Namespace) -> pd.DataFrame:
    out_dir = ensure_dir(args.out_dir)
    rows: list[dict[str, Any]] = []
    model_outputs: dict[str, Any] = {}

    for model_name, (val_df, test_df) in load_default_models(args).items():
        selected = select_threshold(val_df, args.optimize_metric)
        threshold = float(selected["threshold"])
        test_default = binary_metrics(test_df["label"].to_numpy(), test_df["score_lie"].to_numpy(), threshold=0.5)
        test_tuned = binary_metrics(test_df["label"].to_numpy(), test_df["score_lie"].to_numpy(), threshold=threshold)
        test_tuned["bootstrap_95ci"] = bootstrap_ci(
            test_df["label"].to_numpy(),
            test_df["score_lie"].to_numpy(),
            seed=args.seed,
        )

        tuned_pred = test_df.copy()
        tuned_pred["threshold"] = threshold
        tuned_pred["pred_label"] = (tuned_pred["score_lie"] >= threshold).astype(int)
        tuned_pred.to_csv(out_dir / f"{model_name}_test_predictions_tuned.csv", index=False)

        row = {
            "model": model_name,
            "optimize_metric": args.optimize_metric,
            "selected_threshold": threshold,
            "val_selection_score": float(selected["selection_score"]),
            "val_auc_roc": float(selected["metrics"]["auc_roc"]),
            "val_macro_f1_at_threshold": float(selected["metrics"]["macro_f1"]),
            "val_lie_recall_at_threshold": float(selected["metrics"]["recall_lie"]),
            "test_auc_roc": float(test_tuned["auc_roc"]),
            "test_macro_f1_default": float(test_default["macro_f1"]),
            "test_macro_f1_tuned": float(test_tuned["macro_f1"]),
            "test_lie_recall_default": float(test_default["recall_lie"]),
            "test_lie_recall_tuned": float(test_tuned["recall_lie"]),
            "test_accuracy_default": float(test_default["accuracy"]),
            "test_accuracy_tuned": float(test_tuned["accuracy"]),
            "test_eer": float(test_tuned["eer"]),
        }
        rows.append(row)
        model_outputs[model_name] = {
            "threshold_selection": selected,
            "test_default_threshold_metrics": test_default,
            "test_tuned_threshold_metrics": test_tuned,
        }

    df = pd.DataFrame(rows).sort_values("test_macro_f1_tuned", ascending=False)
    df.to_csv(out_dir / f"threshold_tuning_{args.optimize_metric}.csv", index=False)
    write_json(out_dir / f"threshold_tuning_{args.optimize_metric}.json", model_outputs)
    write_markdown(df, out_dir / f"threshold_tuning_{args.optimize_metric}.md")
    return df


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    display = df.copy()
    numeric_cols = [col for col in display.columns if col != "model" and col != "optimize_metric"]
    for col in numeric_cols:
        display[col] = display[col].map(lambda value: f"{float(value):.3f}")
    cols = [
        "model",
        "selected_threshold",
        "val_selection_score",
        "test_auc_roc",
        "test_macro_f1_default",
        "test_macro_f1_tuned",
        "test_lie_recall_default",
        "test_lie_recall_tuned",
        "test_accuracy_tuned",
    ]
    lines = [
        "# Threshold Tuning\n",
        f"Optimized on validation: `{df['optimize_metric'].iloc[0]}`\n",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    lines.append("\nNotes:\n")
    lines.append("- Threshold is selected on validation only, then applied once to test predictions.\n")
    lines.append("- AUC-ROC and EER do not depend on the classification threshold; threshold tuning affects accuracy, F1, recall, precision, and confusion matrix.\n")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune binary classification thresholds on validation predictions.")
    parser.add_argument(
        "--optimize-metric",
        default="macro_f1",
        choices=["macro_f1", "f1_lie", "recall_lie", "youden_j"],
    )
    parser.add_argument("--out-dir", default="outputs/metrics/threshold_tuning")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run(args)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
