from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.evaluate import binary_metrics, find_best_threshold
from src.utils import ensure_dir, write_json


DEFAULT_MODEL_DIRS = {
    "cross_attention_auc": Path("outputs/metrics/dolos_three_stream_auc"),
    "gated_prior_kl": Path("outputs/metrics/dolos_three_stream_gated_logits_prior_kl"),
    "gated_temp_aux": Path("outputs/metrics/dolos_three_stream_gated_logits_temp_aux"),
    "gated_margin_norm": Path("outputs/metrics/dolos_three_stream_gated_margin_norm"),
    "audio_only": Path("outputs/metrics/dolos_fold3_audio_only"),
    "spatial_only": Path("outputs/metrics/dolos_fold3_spatial_only"),
    "spatial_skip1s": Path("outputs/metrics/dolos_fold3_spatial_skip1s"),
    "spatial_clean_faces": Path("outputs/metrics/dolos_fold3_spatial_clean_faces"),
    "flow_only": Path("outputs/metrics/dolos_fold3_flow_only"),
    "flow_clean_faces": Path("outputs/metrics/dolos_fold3_flow_clean_faces"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Val-tuned prediction-level ensemble across existing model predictions.")
    parser.add_argument("--folds", nargs="+", default=["fold1", "fold2", "fold3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="outputs/metrics/prediction_level_ensemble")
    parser.add_argument("--grid-step", type=float, default=0.01)
    parser.add_argument("--threshold-metric", default="balanced_accuracy")
    parser.add_argument(
        "--allow-partial-models",
        action="store_true",
        help="Include models even if they are only available for a subset of folds.",
    )
    parser.add_argument(
        "--only-models",
        nargs="+",
        default=None,
        help="Restrict evaluation to these model names after availability filtering.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Optional model spec name=metrics_dir. If provided, these are added to defaults or override same name.",
    )
    return parser.parse_args()


def parse_model_specs(specs: list[str]) -> dict[str, Path]:
    model_dirs = dict(DEFAULT_MODEL_DIRS)
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--model must be formatted as name=metrics_dir, got: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty model name in --model spec: {spec}")
        model_dirs[name] = Path(path)
    return model_dirs


def prediction_path(model_dir: Path, fold: str, seed: int, split: str) -> Path:
    return model_dir / f"{fold}_seed{seed}_{split}_predictions.csv"


def available_folds(model_dir: Path, folds: list[str], seed: int) -> list[str]:
    available: list[str] = []
    for fold in folds:
        if prediction_path(model_dir, fold, seed, "val").exists() and prediction_path(model_dir, fold, seed, "test").exists():
            available.append(fold)
    return available


def select_models(model_dirs: dict[str, Path], folds: list[str], seed: int, allow_partial: bool) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    availability: list[dict[str, Any]] = []
    selected: dict[str, Path] = {}
    for name, model_dir in model_dirs.items():
        folds_available = available_folds(model_dir, folds, seed)
        availability.append(
            {
                "model": name,
                "metrics_dir": str(model_dir),
                "available_folds": ",".join(folds_available),
                "num_available_folds": len(folds_available),
                "selected": bool(folds_available) if allow_partial else len(folds_available) == len(folds),
            }
        )
        if allow_partial:
            if folds_available:
                selected[name] = model_dir
        elif len(folds_available) == len(folds):
            selected[name] = model_dir
    if len(selected) < 2:
        raise ValueError(f"Need at least two selected models for ensemble, selected={list(selected)}")
    return selected, availability


def restrict_models(model_dirs: dict[str, Path], only_models: list[str] | None) -> dict[str, Path]:
    if not only_models:
        return model_dirs
    missing = [name for name in only_models if name not in model_dirs]
    if missing:
        raise ValueError(f"--only-models contains unknown model names: {missing}. Available: {sorted(model_dirs)}")
    return {name: model_dirs[name] for name in only_models}


def load_predictions(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"video_id", "label", "score_lie"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df[["video_id", "label", "score_lie"]].rename(columns={"score_lie": f"score_{model_name}"})


def load_split(model_dirs: dict[str, Path], fold: str, seed: int, split: str) -> tuple[pd.DataFrame, list[str]]:
    merged: pd.DataFrame | None = None
    loaded_models: list[str] = []
    expected_counts: dict[str, int] = {}
    for model_name, model_dir in model_dirs.items():
        path = prediction_path(model_dir, fold, seed, split)
        if not path.exists():
            continue
        df = load_predictions(path, model_name)
        expected_counts[model_name] = len(df)
        loaded_models.append(model_name)
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=["video_id", "label"], how="inner", validate="one_to_one")
    if merged is None or len(loaded_models) < 2:
        raise ValueError(f"Need at least two prediction files for {fold}/{split}, loaded={loaded_models}")
    if len(set(expected_counts.values())) != 1 or len(merged) != next(iter(expected_counts.values())):
        raise ValueError(f"Prediction merge mismatch for {fold}/{split}: expected={expected_counts}, merged={len(merged)}")
    return merged.sort_values("video_id").reset_index(drop=True), loaded_models


def simplex_weights(num_models: int, step: float) -> np.ndarray:
    if num_models < 2:
        raise ValueError("num_models must be >= 2")
    if step <= 0.0 or step > 1.0:
        raise ValueError("--grid-step must be in (0, 1].")
    levels = int(round(1.0 / step))
    if not np.isclose(levels * step, 1.0):
        raise ValueError("--grid-step must divide 1.0 exactly enough for a stable simplex grid.")
    rows: list[list[float]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            rows.append([*prefix, remaining])
            return
        for value in range(remaining + 1):
            rec([*prefix, value], remaining - value, slots - 1)

    rec([], levels, num_models)
    return np.asarray(rows, dtype=float) / float(levels)


def score_columns(model_names: list[str]) -> list[str]:
    return [f"score_{name}" for name in model_names]


def as_matrix(df: pd.DataFrame, model_names: list[str]) -> np.ndarray:
    return df[score_columns(model_names)].astype(float).to_numpy()


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def weighted_scores(matrix: np.ndarray, weights: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return matrix @ weights
    if mode == "logit":
        return sigmoid(logit(matrix) @ weights)
    raise ValueError(f"Unsupported mode: {mode}")


def thresholded_metrics(y_true: np.ndarray, scores: np.ndarray, threshold_metric: str) -> tuple[dict[str, float | str], dict[str, Any]]:
    calibration = find_best_threshold(y_true, scores, metric=threshold_metric)
    metrics = binary_metrics(y_true, scores, threshold=float(calibration["threshold"]))
    return calibration, metrics


def fast_balanced_accuracy_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[dict[str, float | str], dict[str, Any]]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        calibration: dict[str, float | str] = {"threshold": 0.5, "metric": "balanced_accuracy", "metric_value": 0.0}
        return calibration, {
            "balanced_accuracy": 0.0,
            "f1_lie": 0.0,
            "macro_f1": 0.0,
            "threshold": 0.5,
        }

    order = np.argsort(scores)[::-1]
    sorted_y = y_true[order]
    sorted_scores = scores[order]
    unique_ends = np.r_[np.where(sorted_scores[:-1] != sorted_scores[1:])[0], len(sorted_scores) - 1]

    tp = np.cumsum(sorted_y == 1)[unique_ends].astype(float)
    fp = np.cumsum(sorted_y == 0)[unique_ends].astype(float)
    fn = positives - tp
    tn = negatives - fp
    balanced_accuracy = 0.5 * ((tp / positives) + (tn / negatives))
    f1_lie = np.divide(2.0 * tp, 2.0 * tp + fp + fn, out=np.zeros_like(tp), where=(2.0 * tp + fp + fn) > 0)
    f1_truth = np.divide(2.0 * tn, 2.0 * tn + fp + fn, out=np.zeros_like(tn), where=(2.0 * tn + fp + fn) > 0)
    macro_f1 = 0.5 * (f1_lie + f1_truth)
    best_order = np.lexsort((macro_f1, balanced_accuracy))
    best_idx = int(best_order[-1])
    threshold = float(sorted_scores[unique_ends[best_idx]])
    calibration = {
        "threshold": threshold,
        "metric": "balanced_accuracy",
        "metric_value": float(balanced_accuracy[best_idx]),
    }
    metrics = {
        "balanced_accuracy": float(balanced_accuracy[best_idx]),
        "f1_lie": float(f1_lie[best_idx]),
        "macro_f1": float(macro_f1[best_idx]),
        "threshold": threshold,
    }
    return calibration, metrics


def grid_search_weights(
    val_df: pd.DataFrame,
    model_names: list[str],
    mode: str,
    objective: str,
    grid_step: float,
    threshold_metric: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    y_val = val_df["label"].astype(int).to_numpy()
    x_val = as_matrix(val_df, model_names)
    candidates: list[dict[str, Any]] = []
    weights_grid = simplex_weights(len(model_names), grid_step)
    for weights in weights_grid:
        scores = weighted_scores(x_val, weights, mode)
        auc = float(roc_auc_score(y_val, scores))
        if threshold_metric == "balanced_accuracy":
            calibration, calibrated = fast_balanced_accuracy_threshold(y_val, scores)
        else:
            calibration, calibrated = thresholded_metrics(y_val, scores, threshold_metric)
        objective_value = auc if objective == "auc_roc" else float(calibrated["balanced_accuracy"])
        row: dict[str, Any] = {
            "mode": mode,
            "objective": objective,
            "objective_value": objective_value,
            "val_auc_roc": auc,
            "val_calibrated_threshold": float(calibration["threshold"]),
            "val_calibrated_balanced_accuracy": float(calibrated["balanced_accuracy"]),
            "val_calibrated_f1_lie": float(calibrated["f1_lie"]),
            "val_calibrated_macro_f1": float(calibrated["macro_f1"]),
        }
        row.update({f"w_{name}": float(weight) for name, weight in zip(model_names, weights, strict=True)})
        candidates.append(row)
    grid = pd.DataFrame(candidates)
    best = grid.sort_values(["objective_value", "val_auc_roc", "val_calibrated_macro_f1"], ascending=False).iloc[0].to_dict()
    return best, grid


def evaluate_method(
    fold: str,
    method: str,
    model_names: list[str],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_score: np.ndarray,
    test_score: np.ndarray,
    params: dict[str, Any],
    threshold_metric: str,
) -> dict[str, Any]:
    y_val = val_df["label"].astype(int).to_numpy()
    y_test = test_df["label"].astype(int).to_numpy()
    calibration, val_calibrated = thresholded_metrics(y_val, val_score, threshold_metric)
    threshold = float(calibration["threshold"])
    return {
        "fold": fold,
        "method": method,
        "models": model_names,
        "params": params,
        "threshold_calibration": calibration,
        "val_metrics_default": binary_metrics(y_val, val_score),
        "val_metrics_calibrated": val_calibrated,
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


def row_from_result(result: dict[str, Any], model_names: list[str]) -> dict[str, Any]:
    params = result["params"]
    val = result["val_metrics_calibrated"]
    test = result["test_metrics_calibrated"]
    row: dict[str, Any] = {
        "fold": result["fold"],
        "method": result["method"],
        "threshold": float(result["threshold_calibration"]["threshold"]),
        "val_auc_roc": float(val["auc_roc"]),
        "val_balanced_accuracy": float(val["balanced_accuracy"]),
        "val_f1_lie": float(val["f1_lie"]),
        "val_macro_f1": float(val["macro_f1"]),
        "test_auc_roc": float(test["auc_roc"]),
        "test_balanced_accuracy": float(test["balanced_accuracy"]),
        "test_f1_lie": float(test["f1_lie"]),
        "test_macro_f1": float(test["macro_f1"]),
        "test_confusion_matrix": test["confusion_matrix"],
        "test_score_mean": float(result["score_stats"]["test_mean"]),
        "test_score_std": float(result["score_stats"]["test_std"]),
        "extra": json.dumps({key: value for key, value in params.items() if not key.startswith("w_")}, sort_keys=True),
    }
    for name in model_names:
        row[f"w_{name}"] = params.get(f"w_{name}", "")
    return row


def aggregate_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    summary: list[dict[str, Any]] = []
    for method, group in df.groupby("method", sort=False):
        item: dict[str, Any] = {"fold": "mean", "method": method}
        for key in [
            "val_auc_roc",
            "val_balanced_accuracy",
            "test_auc_roc",
            "test_balanced_accuracy",
            "test_f1_lie",
            "test_macro_f1",
        ]:
            item[key] = float(group[key].astype(float).mean())
        for col in [col for col in df.columns if col.startswith("w_")]:
            values = pd.to_numeric(group[col], errors="coerce")
            item[col] = float(values.mean()) if values.notna().any() else ""
        summary.append(item)
    return summary


def write_rows_csv(path: Path, rows: list[dict[str, Any]], model_names: list[str]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "fold",
        "method",
        *[f"w_{name}" for name in model_names],
        "threshold",
        "val_auc_roc",
        "val_balanced_accuracy",
        "val_f1_lie",
        "val_macro_f1",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(path: Path, rows: list[dict[str, Any]], model_names: list[str]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "fold",
        "method",
        *[f"w_{name}" for name in model_names],
        "val_auc_roc",
        "val_balanced_accuracy",
        "test_auc_roc",
        "test_balanced_accuracy",
        "test_f1_lie",
        "test_macro_f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any) -> str:
    if value == "" or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}"


def weight_text(row: dict[str, Any], model_names: list[str]) -> str:
    parts = []
    for name in model_names:
        value = row.get(f"w_{name}", "")
        if value != "" and not pd.isna(value):
            parts.append(f"{name}={float(value):.2f}")
    return ", ".join(parts) if parts else "-"


def render_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    availability: list[dict[str, Any]],
    selected_models: list[str],
    folds: list[str],
    grid_step: float,
) -> None:
    lines = [
        "# Prediction-Level Ensemble",
        "",
        f"Folds: `{', '.join(folds)}`.",
        f"Grid step: `{grid_step}`.",
        "Weights are tuned on validation per fold, then applied to that fold's test set. Thresholds are calibrated on validation by balanced accuracy.",
        "",
        "## Selected Models",
        "",
        "| Model | Metrics Dir | Folds | Selected |",
        "| --- | --- | --- | --- |",
    ]
    for item in availability:
        selected = "yes" if item["model"] in selected_models else "no"
        lines.append(f"| `{item['model']}` | `{item['metrics_dir']}` | `{item['available_folds']}` | {selected} |")

    lines.extend(
        [
            "",
            "## Mean Across Folds",
            "",
            "| Method | Mean Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {method} | {weights} | {val_auc} | {val_ba} | {test_auc} | {test_ba} | {test_f1} | {test_macro} |".format(
                method=f"`{row['method']}`",
                weights=weight_text(row, selected_models),
                val_auc=pct(row["val_auc_roc"]),
                val_ba=pct(row["val_balanced_accuracy"]),
                test_auc=pct(row["test_auc_roc"]),
                test_ba=pct(row["test_balanced_accuracy"]),
                test_f1=pct(row["test_f1_lie"]),
                test_macro=pct(row["test_macro_f1"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Fold Results",
            "",
            "| Fold | Method | Weights | Val AUC | Val BA | Test AUC | Test BA | Test F1 | Test Macro | Test CM |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {fold} | {method} | {weights} | {val_auc} | {val_ba} | {test_auc} | {test_ba} | {test_f1} | {test_macro} | `{cm}` |".format(
                fold=row["fold"],
                method=f"`{row['method']}`",
                weights=weight_text(row, selected_models),
                val_auc=pct(row["val_auc_roc"]),
                val_ba=pct(row["val_balanced_accuracy"]),
                test_auc=pct(row["test_auc_roc"]),
                test_ba=pct(row["test_balanced_accuracy"]),
                test_f1=pct(row["test_f1_lie"]),
                test_macro=pct(row["test_macro_f1"]),
                cm=row["test_confusion_matrix"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This report only compares models that have both validation and test predictions for the selected folds.",
            "- The main comparison target is `gated_prior_kl`, because it is the current primary model.",
            "- A useful ensemble should improve mean test AUC or mean calibrated BA without relying on a single fold spike.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    folds = list(args.folds)
    out_dir = ensure_dir(args.out_dir)
    all_model_dirs = parse_model_specs(args.model)
    all_model_dirs = restrict_models(all_model_dirs, args.only_models)
    model_dirs, availability = select_models(all_model_dirs, folds, args.seed, args.allow_partial_models)
    selected_names = list(model_dirs)

    all_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for fold in folds:
        val_df, val_models = load_split(model_dirs, fold, args.seed, "val")
        test_df, test_models = load_split(model_dirs, fold, args.seed, "test")
        model_names = [name for name in selected_names if name in val_models and name in test_models]
        if model_names != val_models or model_names != test_models:
            val_df = val_df[["video_id", "label", *score_columns(model_names)]]
            test_df = test_df[["video_id", "label", *score_columns(model_names)]]

        x_val = as_matrix(val_df, model_names)
        x_test = as_matrix(test_df, model_names)
        fold_dir = ensure_dir(out_dir / fold)

        for idx, model_name in enumerate(model_names):
            val_score = x_val[:, idx]
            test_score = x_test[:, idx]
            params = {f"w_{name}": 1.0 if name == model_name else 0.0 for name in model_names}
            params["source"] = "single_model"
            result = evaluate_method(fold, model_name, model_names, val_df, test_df, val_score, test_score, params, args.threshold_metric)
            all_results.append(result)
            rows.append(row_from_result(result, selected_names))

        for mode in ["raw", "logit"]:
            for objective in ["balanced_accuracy", "auc_roc"]:
                best, grid = grid_search_weights(
                    val_df,
                    model_names,
                    mode=mode,
                    objective=objective,
                    grid_step=float(args.grid_step),
                    threshold_metric=args.threshold_metric,
                )
                weights = np.asarray([float(best[f"w_{name}"]) for name in model_names], dtype=float)
                method = f"ensemble_{mode}_{objective}"
                val_score = weighted_scores(x_val, weights, mode)
                test_score = weighted_scores(x_test, weights, mode)
                params = {
                    "mode": mode,
                    "objective": objective,
                    "grid_step": float(args.grid_step),
                    **{f"w_{name}": float(weight) for name, weight in zip(model_names, weights, strict=True)},
                }
                result = evaluate_method(fold, method, model_names, val_df, test_df, val_score, test_score, params, args.threshold_metric)
                all_results.append(result)
                rows.append(row_from_result(result, selected_names))
                grid.sort_values(["objective_value", "val_auc_roc", "val_calibrated_macro_f1"], ascending=False).head(100).to_csv(
                    fold_dir / f"{method}_top100_grid.csv",
                    index=False,
                )
                prediction_frame(val_df, val_score, method).to_csv(fold_dir / f"{method}_val_predictions.csv", index=False)
                prediction_frame(test_df, test_score, method).to_csv(fold_dir / f"{method}_test_predictions.csv", index=False)

    summary_rows = aggregate_summary(rows)
    write_json(out_dir / "prediction_level_ensemble_results.json", all_results)
    pd.DataFrame(availability).to_csv(out_dir / "model_availability.csv", index=False)
    write_rows_csv(out_dir / "prediction_level_ensemble_results.csv", rows, selected_names)
    write_summary_csv(out_dir / "prediction_level_ensemble_summary.csv", summary_rows, selected_names)
    render_markdown(
        out_dir / "prediction_level_ensemble_results.md",
        rows,
        summary_rows,
        availability,
        selected_names,
        folds,
        float(args.grid_step),
    )
    print(out_dir / "prediction_level_ensemble_results.md")
    for row in summary_rows:
        print(
            row["method"],
            "test_auc",
            f"{float(row['test_auc_roc']):.4f}",
            "test_ba",
            f"{float(row['test_balanced_accuracy']):.4f}",
            "weights",
            weight_text(row, selected_names),
        )


if __name__ == "__main__":
    main()
