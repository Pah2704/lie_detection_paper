from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import binary_metrics, find_best_threshold
from src.utils import ensure_dir, write_json


STREAM_DIRS = {
    "spatial": Path("outputs/metrics/dolos_fold3_spatial_only"),
    "flow": Path("outputs/metrics/dolos_fold3_flow_only"),
    "audio": Path("outputs/metrics/dolos_fold3_audio_only"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse fold stream predictions with val-tuned weights.")
    parser.add_argument("--fold", default="fold3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/metrics/fold3_stream_fusion")
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--threshold-metric", default="balanced_accuracy")
    return parser.parse_args()


def load_split(fold: str, seed: int, split: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for stream, metrics_dir in STREAM_DIRS.items():
        path = metrics_dir / f"{fold}_seed{seed}_{split}_predictions.csv"
        df = pd.read_csv(path)[["video_id", "label", "score_lie"]].rename(columns={"score_lie": f"score_{stream}"})
        if merged is None:
            merged = df
            continue
        merged = merged.merge(df, on=["video_id", "label"], how="inner", validate="one_to_one")
    if merged is None:
        raise ValueError("No stream predictions were loaded.")
    expected = {stream: len(pd.read_csv(metrics_dir / f"{fold}_seed{seed}_{split}_predictions.csv")) for stream, metrics_dir in STREAM_DIRS.items()}
    if len(set(expected.values())) != 1 or len(merged) != next(iter(expected.values())):
        raise ValueError(f"Prediction merge mismatch for {split}: expected={expected}, merged={len(merged)}")
    return merged.sort_values("video_id").reset_index(drop=True)


def simplex_weights(step: float) -> list[tuple[float, float, float]]:
    if step <= 0 or step > 1:
        raise ValueError("--grid-step must be in (0, 1].")
    levels = int(round(1.0 / step))
    weights: list[tuple[float, float, float]] = []
    for i in range(levels + 1):
        for j in range(levels + 1 - i):
            k = levels - i - j
            weights.append((i / levels, j / levels, k / levels))
    return weights


def as_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[["score_spatial", "score_flow", "score_audio"]].astype(float).to_numpy()


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def weighted_scores(matrix: np.ndarray, weights: tuple[float, float, float], mode: str) -> np.ndarray:
    weight_vec = np.asarray(weights, dtype=float)
    if mode == "raw":
        return matrix @ weight_vec
    if mode == "logit":
        return sigmoid(logit(matrix) @ weight_vec)
    raise ValueError(f"Unsupported weighted score mode: {mode}")


def best_balanced_accuracy_threshold(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return {"threshold": 0.5, "balanced_accuracy": 0.0, "f1_lie": 0.0, "macro_f1": 0.0}

    order = np.argsort(y_score)[::-1]
    sorted_y = y_true[order]
    sorted_score = y_score[order]
    unique_ends = np.r_[np.where(sorted_score[:-1] != sorted_score[1:])[0], len(sorted_score) - 1]

    tp = np.cumsum(sorted_y == 1)[unique_ends].astype(float)
    fp = np.cumsum(sorted_y == 0)[unique_ends].astype(float)
    fn = positives - tp
    tn = negatives - fp
    balanced_accuracy = 0.5 * ((tp / positives) + (tn / negatives))
    f1_lie = np.divide(2.0 * tp, 2.0 * tp + fp + fn, out=np.zeros_like(tp), where=(2.0 * tp + fp + fn) > 0)
    f1_truth = np.divide(2.0 * tn, 2.0 * tn + fp + fn, out=np.zeros_like(tn), where=(2.0 * tn + fp + fn) > 0)
    macro_f1 = 0.5 * (f1_lie + f1_truth)

    # Tie break toward less collapsed thresholds.
    best_order = np.lexsort((macro_f1, balanced_accuracy))
    best_idx = int(best_order[-1])
    return {
        "threshold": float(sorted_score[unique_ends[best_idx]]),
        "balanced_accuracy": float(balanced_accuracy[best_idx]),
        "f1_lie": float(f1_lie[best_idx]),
        "macro_f1": float(macro_f1[best_idx]),
    }


def select_weighted_fusion(
    val_df: pd.DataFrame,
    mode: str,
    objective: str,
    grid_step: float,
    threshold_metric: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_val = val_df["label"].astype(int).to_numpy()
    x_val = as_matrix(val_df)
    candidates: list[dict[str, Any]] = []
    for weights in simplex_weights(grid_step):
        scores = weighted_scores(x_val, weights, mode)
        auc = float(roc_auc_score(y_val, scores))
        if threshold_metric == "balanced_accuracy":
            fast_metrics = best_balanced_accuracy_threshold(y_val, scores)
            threshold = float(fast_metrics["threshold"])
            calibrated_ba = float(fast_metrics["balanced_accuracy"])
            calibrated_f1 = float(fast_metrics["f1_lie"])
            calibrated_macro = float(fast_metrics["macro_f1"])
        else:
            calibration = find_best_threshold(y_val, scores, metric=threshold_metric)
            metrics_at_calibrated = binary_metrics(y_val, scores, threshold=float(calibration["threshold"]))
            threshold = float(calibration["threshold"])
            calibrated_ba = float(metrics_at_calibrated["balanced_accuracy"])
            calibrated_f1 = float(metrics_at_calibrated["f1_lie"])
            calibrated_macro = float(metrics_at_calibrated["macro_f1"])
        objective_value = auc if objective == "auc_roc" else calibrated_ba
        candidates.append(
            {
                "w_spatial": weights[0],
                "w_flow": weights[1],
                "w_audio": weights[2],
                "objective": objective,
                "objective_value": objective_value,
                "val_auc_roc": auc,
                "val_calibrated_threshold": threshold,
                "val_calibrated_balanced_accuracy": calibrated_ba,
                "val_calibrated_f1_lie": calibrated_f1,
                "val_calibrated_macro_f1": calibrated_macro,
            }
        )
    grid = pd.DataFrame(candidates)
    sort_cols = ["objective_value", "val_auc_roc", "val_calibrated_macro_f1"]
    best_row = grid.sort_values(sort_cols, ascending=False).iloc[0].to_dict()
    return best_row, grid


def evaluate_scores(
    name: str,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_score: np.ndarray,
    test_score: np.ndarray,
    params: dict[str, Any],
    threshold_metric: str,
) -> dict[str, Any]:
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()
    calibration = find_best_threshold(y_val, val_score, metric=threshold_metric)
    threshold = float(calibration["threshold"])
    return {
        "method": name,
        "params": params,
        "threshold_calibration": calibration,
        "val_metrics_default": binary_metrics(y_val, val_score),
        "val_metrics_calibrated": binary_metrics(y_val, val_score, threshold=threshold),
        "test_metrics_default": binary_metrics(y_test, test_score),
        "test_metrics_calibrated": binary_metrics(y_test, test_score, threshold=threshold),
        "score_stats": {
            "val_mean": float(np.mean(val_score)),
            "val_std": float(np.std(val_score)),
            "test_mean": float(np.mean(test_score)),
            "test_std": float(np.std(test_score)),
        },
    }


def prediction_frame(base: pd.DataFrame, score: np.ndarray, method: str) -> pd.DataFrame:
    out = base[["video_id", "label"]].copy()
    out["score_lie"] = score.astype(float)
    out["fusion_method"] = method
    return out


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "method",
        "w_spatial",
        "w_flow",
        "w_audio",
        "threshold",
        "val_auc_roc",
        "val_balanced_accuracy",
        "val_f1_lie",
        "test_auc_roc",
        "test_balanced_accuracy",
        "test_f1_lie",
        "test_macro_f1",
        "test_confusion_matrix",
        "test_score_mean",
        "test_score_std",
        "extra",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def method_row(result: dict[str, Any]) -> dict[str, Any]:
    params = result["params"]
    val = result["val_metrics_calibrated"]
    test = result["test_metrics_calibrated"]
    return {
        "method": result["method"],
        "w_spatial": params.get("w_spatial", ""),
        "w_flow": params.get("w_flow", ""),
        "w_audio": params.get("w_audio", ""),
        "threshold": result["threshold_calibration"]["threshold"],
        "val_auc_roc": val["auc_roc"],
        "val_balanced_accuracy": val["balanced_accuracy"],
        "val_f1_lie": val["f1_lie"],
        "test_auc_roc": test["auc_roc"],
        "test_balanced_accuracy": test["balanced_accuracy"],
        "test_f1_lie": test["f1_lie"],
        "test_macro_f1": test["macro_f1"],
        "test_confusion_matrix": test["confusion_matrix"],
        "test_score_mean": result["score_stats"]["test_mean"],
        "test_score_std": result["score_stats"]["test_std"],
        "extra": json.dumps({key: value for key, value in params.items() if key not in {"w_spatial", "w_flow", "w_audio"}}, sort_keys=True),
    }


def render_markdown(path: Path, rows: list[dict[str, Any]], grid_step: float) -> None:
    def pct(value: float) -> str:
        return f"{value * 100:.2f}"

    def num(value: float) -> str:
        return f"{value:.4f}"

    lines = [
        "# Fold3 Prediction Fusion",
        "",
        f"Grid step: `{grid_step}`. Thresholds are selected on validation by balanced accuracy.",
        "",
        "| Method | Weights S/F/A | Thr | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM | Test score mean/std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        weights = f"{row['w_spatial']}/{row['w_flow']}/{row['w_audio']}" if row["w_spatial"] != "" else "-"
        lines.append(
            "| {method} | {weights} | {thr} | {val_auc} | {val_ba} | {test_auc} | {test_ba} | {test_f1} | {test_macro} | `{cm}` | {score} |".format(
                method=row["method"],
                weights=weights,
                thr=num(float(row["threshold"])),
                val_auc=pct(float(row["val_auc_roc"])),
                val_ba=pct(float(row["val_balanced_accuracy"])),
                test_auc=pct(float(row["test_auc_roc"])),
                test_ba=pct(float(row["test_balanced_accuracy"])),
                test_f1=pct(float(row["test_f1_lie"])),
                test_macro=pct(float(row["test_macro_f1"])),
                cm=row["test_confusion_matrix"],
                score=f"{float(row['test_score_mean']):.4f}/{float(row['test_score_std']):.4f}",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The val-tuned fusion improves validation BA, but fold3 test BA remains modest.",
            "- If best weights collapse to one stream, the other streams are not adding useful validation signal under this objective.",
            "- Compare test AUC with single-stream and full-model summaries before running full 3-fold.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    val_df = load_split(args.fold, args.seed, "val")
    test_df = load_split(args.fold, args.seed, "test")
    val_matrix = as_matrix(val_df)
    test_matrix = as_matrix(test_df)

    results: list[dict[str, Any]] = []
    for mode in ["raw", "logit"]:
        for objective in ["balanced_accuracy", "auc_roc"]:
            best, grid = select_weighted_fusion(
                val_df,
                mode=mode,
                objective=objective,
                grid_step=float(args.grid_step),
                threshold_metric=args.threshold_metric,
            )
            method = f"late_{mode}_{objective}"
            weights = (float(best["w_spatial"]), float(best["w_flow"]), float(best["w_audio"]))
            val_score = weighted_scores(val_matrix, weights, mode)
            test_score = weighted_scores(test_matrix, weights, mode)
            params = {
                "mode": mode,
                "objective": objective,
                "w_spatial": weights[0],
                "w_flow": weights[1],
                "w_audio": weights[2],
                "grid_step": float(args.grid_step),
            }
            result = evaluate_scores(method, val_df, test_df, val_score, test_score, params, args.threshold_metric)
            results.append(result)
            grid.sort_values(["objective_value", "val_auc_roc", "val_calibrated_macro_f1"], ascending=False).head(50).to_csv(
                out_dir / f"{method}_top50_grid.csv",
                index=False,
            )
            prediction_frame(val_df, val_score, method).to_csv(out_dir / f"{method}_val_predictions.csv", index=False)
            prediction_frame(test_df, test_score, method).to_csv(out_dir / f"{method}_test_predictions.csv", index=False)

    gate = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed)),
        ]
    )
    y_val = val_df["label"].astype(int).to_numpy()
    gate.fit(val_matrix, y_val)
    val_gate = gate.predict_proba(val_matrix)[:, 1]
    test_gate = gate.predict_proba(test_matrix)[:, 1]
    clf = gate.named_steps["clf"]
    params = {
        "model": "StandardScaler+LogisticRegression",
        "class_weight": "balanced",
        "coef_spatial": float(clf.coef_[0][0]),
        "coef_flow": float(clf.coef_[0][1]),
        "coef_audio": float(clf.coef_[0][2]),
        "intercept": float(clf.intercept_[0]),
    }
    gated = evaluate_scores("gated_logreg", val_df, test_df, val_gate, test_gate, params, args.threshold_metric)
    results.append(gated)
    prediction_frame(val_df, val_gate, "gated_logreg").to_csv(out_dir / "gated_logreg_val_predictions.csv", index=False)
    prediction_frame(test_df, test_gate, "gated_logreg").to_csv(out_dir / "gated_logreg_test_predictions.csv", index=False)

    rows = [method_row(result) for result in results]
    write_json(out_dir / "fold3_prediction_fusion_results.json", results)
    write_table(out_dir / "fold3_prediction_fusion_results.csv", rows)
    render_markdown(out_dir / "fold3_prediction_fusion_results.md", rows, grid_step=float(args.grid_step))
    print(out_dir / "fold3_prediction_fusion_results.md")
    for row in rows:
        print(
            row["method"],
            "weights",
            row["w_spatial"],
            row["w_flow"],
            row["w_audio"],
            "test_auc",
            f"{float(row['test_auc_roc']):.4f}",
            "test_ba",
            f"{float(row['test_balanced_accuracy']):.4f}",
        )


if __name__ == "__main__":
    main()
