from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

from src.evaluate import binary_metrics
from src.utils import ensure_dir


PAPER_SOURCE = "https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_Audio-Visual_Deception_Detection_DOLOS_Dataset_and_Parameter-Efficient_Crossmodal_Learning_ICCV_2023_paper.pdf"
PAPER_ROWS = [
    {"method": "DOLOS paper Visual", "acc": 61.44, "f1_lie": 69.42, "auc_roc": 58.89, "source": "Table 2"},
    {"method": "DOLOS paper Audio", "acc": 59.19, "f1_lie": 73.46, "auc_roc": 52.54, "source": "Table 2"},
    {"method": "DOLOS paper Concatenation", "acc": 61.62, "f1_lie": 70.20, "auc_roc": 60.50, "source": "Table 2"},
    {"method": "DOLOS paper PAVF", "acc": 64.75, "f1_lie": 71.20, "auc_roc": 62.71, "source": "Table 2"},
    {"method": "DOLOS paper PAVF + Multi-task", "acc": 66.84, "f1_lie": 73.35, "auc_roc": 64.58, "source": "Table 2"},
]


METHOD_LABELS = {
    "cross_attention_auc": "Cross-attention AUC baseline",
    "gated_prior_kl": "Gated logits prior-KL",
    "ensemble_raw_auc_roc": "Final ensemble raw-AUC",
    "ensemble_raw_balanced_accuracy": "Final ensemble raw-BA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final report tables and error analysis for the selected ensemble.")
    parser.add_argument("--ensemble-dir", default="outputs/metrics/prediction_level_ensemble")
    parser.add_argument("--metadata", default="data/processed/splits/dolos/cache_filtered/metadata.csv")
    parser.add_argument("--contamination-audit", default="outputs/metrics/face_contamination_audit/clip_face_contamination_audit.csv")
    parser.add_argument("--out-dir", default="outputs/metrics/final_report")
    parser.add_argument("--final-method", default="ensemble_raw_auc_roc")
    return parser.parse_args()


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}"


def pct_plain(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def load_results(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def weights_text(params: dict[str, Any]) -> str:
    weights = []
    for key, value in params.items():
        if key.startswith("w_"):
            weights.append((key.removeprefix("w_"), float(value)))
    if not weights:
        return ""
    return ";".join(f"{name}={weight:.2f}" for name, weight in weights)


def per_fold_table(results: list[dict[str, Any]], methods: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        method = result["method"]
        if method not in methods:
            continue
        default = result["test_metrics_default"]
        calibrated = result["test_metrics_calibrated"]
        rows.append(
            {
                "fold": result["fold"],
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "weights": weights_text(result.get("params", {})),
                "threshold": calibrated["threshold"],
                "accuracy_at_0p5": default["accuracy"],
                "balanced_accuracy_at_0p5": default["balanced_accuracy"],
                "f1_lie_at_0p5": default["f1_lie"],
                "macro_f1_at_0p5": default["macro_f1"],
                "auc_roc": default["auc_roc"],
                "auc_pr": default["auc_pr"],
                "calibrated_accuracy": calibrated["accuracy"],
                "calibrated_balanced_accuracy": calibrated["balanced_accuracy"],
                "calibrated_precision_lie": calibrated["precision_lie"],
                "calibrated_recall_lie": calibrated["recall_lie"],
                "calibrated_f1_lie": calibrated["f1_lie"],
                "calibrated_macro_f1": calibrated["macro_f1"],
                "calibrated_confusion_matrix": calibrated["confusion_matrix"],
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "fold"]).reset_index(drop=True)


def mean_table(per_fold: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy_at_0p5",
        "balanced_accuracy_at_0p5",
        "f1_lie_at_0p5",
        "macro_f1_at_0p5",
        "auc_roc",
        "auc_pr",
        "calibrated_accuracy",
        "calibrated_balanced_accuracy",
        "calibrated_precision_lie",
        "calibrated_recall_lie",
        "calibrated_f1_lie",
        "calibrated_macro_f1",
    ]
    rows = []
    for method, group in per_fold.groupby("method", sort=False):
        weight_values: dict[str, list[float]] = {}
        for text in group["weights"]:
            for part in str(text).split(";"):
                if "=" not in part:
                    continue
                name, value = part.split("=", 1)
                weight_values.setdefault(name, []).append(float(value))
        mean_weights = ";".join(f"{name}={np.mean(values):.2f}" for name, values in weight_values.items())
        row = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "fold": "mean",
            "weights": mean_weights,
        }
        for metric in metrics:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paper_comparison_table(means: pd.DataFrame, final_method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in PAPER_ROWS:
        rows.append(
            {
                "source_group": "paper",
                "method": item["method"],
                "acc": item["acc"] / 100.0,
                "f1_lie": item["f1_lie"] / 100.0,
                "auc_roc": item["auc_roc"] / 100.0,
                "balanced_accuracy": np.nan,
                "notes": item["source"],
            }
        )
    for method in ["cross_attention_auc", "gated_prior_kl", final_method]:
        row = means[means["method"] == method].iloc[0]
        rows.append(
            {
                "source_group": "ours",
                "method": row["method_label"],
                "acc": row["calibrated_accuracy"],
                "f1_lie": row["calibrated_f1_lie"],
                "auc_roc": row["auc_roc"],
                "balanced_accuracy": row["calibrated_balanced_accuracy"],
                "notes": "mean over fold calibrated thresholds",
            }
        )
    return pd.DataFrame(rows)


def load_final_predictions(ensemble_dir: Path, results: list[dict[str, Any]], final_method: str, metadata_path: Path) -> pd.DataFrame:
    thresholds = {
        result["fold"]: float(result["test_metrics_calibrated"]["threshold"])
        for result in results
        if result["method"] == final_method
    }
    frames = []
    for fold, threshold in thresholds.items():
        path = ensemble_dir / fold / f"{final_method}_test_predictions.csv"
        df = pd.read_csv(path)
        df["fold"] = fold
        df["threshold"] = threshold
        frames.append(df)
    preds = pd.concat(frames, ignore_index=True)
    preds["label"] = preds["label"].astype(int)
    preds["pred"] = (preds["score_lie"] >= preds["threshold"]).astype(int)
    preds["correct"] = preds["pred"] == preds["label"]
    preds["margin"] = (preds["score_lie"] - preds["threshold"]).abs()
    preds["error_type"] = np.select(
        [
            (preds["label"] == 0) & (preds["pred"] == 0),
            (preds["label"] == 0) & (preds["pred"] == 1),
            (preds["label"] == 1) & (preds["pred"] == 0),
            (preds["label"] == 1) & (preds["pred"] == 1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="unknown",
    )
    metadata = pd.read_csv(metadata_path)
    metadata_cols = [
        col
        for col in ["video_id", "host", "episode", "episode_group_id", "group_id", "file_name", "start_time", "end_time"]
        if col in metadata.columns
    ]
    preds = preds.merge(metadata[metadata_cols].drop_duplicates("video_id"), on="video_id", how="left", validate="many_to_one")
    if "episode_group_id" not in preds.columns and "group_id" in preds.columns:
        preds["episode_group_id"] = preds["group_id"]
    return preds


def add_contamination(preds: pd.DataFrame, audit_path: Path) -> pd.DataFrame:
    if not audit_path.exists():
        preds["clip_suspect_contamination"] = np.nan
        preds["first_window_suspect"] = np.nan
        preds["suspect_window_ratio"] = np.nan
        return preds
    audit = pd.read_csv(audit_path)
    cols = [
        col
        for col in [
            "fold",
            "video_id",
            "clip_suspect_contamination",
            "first_window_suspect",
            "suspect_window_ratio",
            "max_identity_clusters",
            "min_dominant_ratio",
            "max_cosine_distance",
        ]
        if col in audit.columns
    ]
    return preds.merge(audit[cols], on=["fold", "video_id"], how="left", validate="one_to_one")


def group_metrics(df: pd.DataFrame, group_col: str, min_n: int = 1) -> pd.DataFrame:
    rows = []
    for group, item in df.groupby(group_col, dropna=False):
        if len(item) < min_n:
            continue
        y_true = item["label"].astype(int).to_numpy()
        y_pred = item["pred"].astype(int).to_numpy()
        y_score = item["score_lie"].astype(float).to_numpy()
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).astype(int)
        tn, fp, fn, tp = cm.ravel()
        auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else np.nan
        rows.append(
            {
                group_col: group,
                "n": int(len(item)),
                "truth": int((item["label"] == 0).sum()),
                "lie": int((item["label"] == 1).sum()),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "f1_lie": float(f1_score(y_true, y_pred, zero_division=0)),
                "auc_roc": auc,
                "pred_lie_rate": float(y_pred.mean()),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    return pd.DataFrame(rows).sort_values(["balanced_accuracy", "n"], ascending=[True, False]).reset_index(drop=True)


def contamination_metrics(preds: pd.DataFrame) -> pd.DataFrame:
    if "clip_suspect_contamination" not in preds.columns:
        return pd.DataFrame()
    rows = []
    for value, item in preds.dropna(subset=["clip_suspect_contamination"]).groupby("clip_suspect_contamination"):
        y_true = item["label"].astype(int).to_numpy()
        y_pred = item["pred"].astype(int).to_numpy()
        auc = float(roc_auc_score(y_true, item["score_lie"].astype(float).to_numpy())) if len(np.unique(y_true)) == 2 else np.nan
        accuracy = float(accuracy_score(y_true, y_pred))
        balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
        f1_lie = float(f1_score(y_true, y_pred, zero_division=0))
        rows.append(
            {
                "clip_suspect_contamination": bool(value),
                "n": int(len(item)),
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1_lie": f1_lie,
                "auc_roc": auc,
                "error_rate": float(1.0 - accuracy),
                "mean_score_lie": float(item["score_lie"].mean()),
            }
        )
    return pd.DataFrame(rows)


def table_lines(df: pd.DataFrame, columns: list[str], format_percent: set[str] | None = None, limit: int | None = None) -> list[str]:
    format_percent = format_percent or set()
    view = df.head(limit) if limit else df
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in view.iterrows():
        cells = []
        for col in columns:
            value = row[col]
            if col in format_percent:
                cells.append(pct(value))
            elif isinstance(value, float):
                cells.append(f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def render_summary(
    path: Path,
    means: pd.DataFrame,
    per_fold: pd.DataFrame,
    paper: pd.DataFrame,
    final_method: str,
    host_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    contamination: pd.DataFrame,
) -> None:
    final = means[means["method"] == final_method].iloc[0]
    gated = means[means["method"] == "gated_prior_kl"].iloc[0]
    auc_delta = float(final["auc_roc"]) - float(gated["auc_roc"])
    ba_delta = float(final["calibrated_balanced_accuracy"]) - float(gated["calibrated_balanced_accuracy"])
    lines = [
        "# Final DOLOS Results",
        "",
        "## Final Decision",
        "",
        f"- Final method: `{final_method}`.",
        "- It is a validation-tuned prediction-level ensemble of `cross_attention_auc` and `gated_prior_kl`.",
        f"- Mean AUC changes from {pct(gated['auc_roc'])} to {pct(final['auc_roc'])} ({auc_delta * 100:+.2f} points).",
        f"- Mean calibrated BA changes from {pct(gated['calibrated_balanced_accuracy'])} to {pct(final['calibrated_balanced_accuracy'])} ({ba_delta * 100:+.2f} points).",
        "",
        "## Mean 3-Fold Metrics",
        "",
    ]
    mean_cols = ["method_label", "auc_roc", "balanced_accuracy_at_0p5", "calibrated_balanced_accuracy", "calibrated_f1_lie", "calibrated_macro_f1"]
    lines.extend(table_lines(means, mean_cols, set(mean_cols) - {"method_label"}))
    lines.extend(
        [
            "",
            "## Final Ensemble Per Fold",
            "",
        ]
    )
    final_folds = per_fold[per_fold["method"] == final_method]
    fold_cols = ["fold", "weights", "threshold", "auc_roc", "balanced_accuracy_at_0p5", "calibrated_balanced_accuracy", "calibrated_f1_lie", "calibrated_confusion_matrix"]
    lines.extend(table_lines(final_folds, fold_cols, {"auc_roc", "balanced_accuracy_at_0p5", "calibrated_balanced_accuracy", "calibrated_f1_lie"}))
    lines.extend(
        [
            "",
            "## DOLOS Paper Comparison",
            "",
            f"Paper source: {PAPER_SOURCE}",
            "",
            "The DOLOS paper reports ACC, F1, and AUC for the official 3-fold average. BA is not reported in the paper, so BA below is only for our models.",
            "",
        ]
    )
    paper_cols = ["source_group", "method", "acc", "balanced_accuracy", "f1_lie", "auc_roc", "notes"]
    lines.extend(table_lines(paper, paper_cols, {"acc", "balanced_accuracy", "f1_lie", "auc_roc"}))
    lines.extend(
        [
            "",
            "## Error Analysis Highlights",
            "",
            "### Worst Hosts By Calibrated BA",
            "",
        ]
    )
    host_cols = ["host", "n", "truth", "lie", "balanced_accuracy", "accuracy", "auc_roc", "pred_lie_rate", "tn", "fp", "fn", "tp"]
    lines.extend(table_lines(host_metrics, host_cols, {"balanced_accuracy", "accuracy", "auc_roc", "pred_lie_rate"}, limit=8))
    lines.extend(["", "### Worst Episodes By Calibrated BA", ""])
    episode_cols = ["episode_group_id", "n", "truth", "lie", "balanced_accuracy", "accuracy", "auc_roc", "pred_lie_rate", "tn", "fp", "fn", "tp"]
    lines.extend(table_lines(episode_metrics, episode_cols, {"balanced_accuracy", "accuracy", "auc_roc", "pred_lie_rate"}, limit=12))
    if not contamination.empty:
        lines.extend(["", "### Face Contamination Heuristic", ""])
        contam_cols = ["clip_suspect_contamination", "n", "balanced_accuracy", "accuracy", "auc_roc", "error_rate", "mean_score_lie"]
        lines.extend(table_lines(contamination, contam_cols, {"balanced_accuracy", "accuracy", "auc_roc", "error_rate", "mean_score_lie"}))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The final ensemble beats both constituent models on mean AUC and calibrated BA.",
            "- The strongest single model remains `gated_prior_kl`.",
            "- Adding standalone audio/spatial predictions was tested separately and rejected because validation tuning did not generalize as well as the 2-model ensemble.",
            "- Error concentration by host/episode remains material, so report conclusions should emphasize protocol-level average performance and residual group sensitivity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_error_markdown(
    path: Path,
    preds: pd.DataFrame,
    host_metrics: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    contamination: pd.DataFrame,
    top_fp: pd.DataFrame,
    top_fn: pd.DataFrame,
) -> None:
    overall_ba = balanced_accuracy_score(preds["label"], preds["pred"])
    overall_acc = accuracy_score(preds["label"], preds["pred"])
    lines = [
        "# Final Ensemble Error Analysis",
        "",
        f"- Clips: {len(preds)}",
        f"- Accuracy: {pct(overall_acc)}",
        f"- Balanced accuracy: {pct(overall_ba)}",
        f"- Error rate: {pct(1.0 - overall_acc)}",
        "",
        "## Error Counts",
        "",
    ]
    counts = preds["error_type"].value_counts().rename_axis("error_type").reset_index(name="n")
    lines.extend(table_lines(counts, ["error_type", "n"]))
    lines.extend(["", "## Worst Hosts", ""])
    host_cols = ["host", "n", "balanced_accuracy", "accuracy", "auc_roc", "tn", "fp", "fn", "tp"]
    lines.extend(table_lines(host_metrics, host_cols, {"balanced_accuracy", "accuracy", "auc_roc"}, limit=8))
    lines.extend(["", "## Worst Episodes", ""])
    episode_cols = ["episode_group_id", "n", "balanced_accuracy", "accuracy", "auc_roc", "tn", "fp", "fn", "tp"]
    lines.extend(table_lines(episode_metrics, episode_cols, {"balanced_accuracy", "accuracy", "auc_roc"}, limit=12))
    if not contamination.empty:
        lines.extend(["", "## Contamination Heuristic", ""])
        contam_cols = ["clip_suspect_contamination", "n", "balanced_accuracy", "accuracy", "auc_roc", "error_rate", "mean_score_lie"]
        lines.extend(table_lines(contamination, contam_cols, {"balanced_accuracy", "accuracy", "auc_roc", "error_rate", "mean_score_lie"}))
    lines.extend(["", "## Most Confident False Positives", ""])
    error_cols = ["fold", "video_id", "host", "episode_group_id", "score_lie", "threshold", "margin"]
    lines.extend(table_lines(top_fp, error_cols, limit=12))
    lines.extend(["", "## Most Confident False Negatives", ""])
    lines.extend(table_lines(top_fn, error_cols, limit=12))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    ensemble_dir = Path(args.ensemble_dir)
    results = load_results(ensemble_dir / "prediction_level_ensemble_results.json")
    methods = {"cross_attention_auc", "gated_prior_kl", "ensemble_raw_auc_roc", "ensemble_raw_balanced_accuracy"}
    per_fold = per_fold_table(results, methods)
    means = mean_table(per_fold)
    paper = paper_comparison_table(means, args.final_method)
    preds = load_final_predictions(ensemble_dir, results, args.final_method, Path(args.metadata))
    preds = add_contamination(preds, Path(args.contamination_audit))

    host_metrics = group_metrics(preds, "host", min_n=20)
    episode_metrics = group_metrics(preds, "episode_group_id", min_n=8)
    contamination = contamination_metrics(preds)
    top_fp = preds[preds["error_type"] == "FP"].sort_values("margin", ascending=False).head(30)
    top_fn = preds[preds["error_type"] == "FN"].sort_values("margin", ascending=False).head(30)

    per_fold.to_csv(out_dir / "final_per_fold_metrics.csv", index=False)
    means.to_csv(out_dir / "final_results_table.csv", index=False)
    paper.to_csv(out_dir / "dolos_paper_comparison.csv", index=False)
    preds.to_csv(out_dir / "final_ensemble_predictions_with_errors.csv", index=False)
    host_metrics.to_csv(out_dir / "final_error_by_host.csv", index=False)
    episode_metrics.to_csv(out_dir / "final_error_by_episode.csv", index=False)
    contamination.to_csv(out_dir / "final_error_by_contamination.csv", index=False)
    top_fp.to_csv(out_dir / "final_top_false_positives.csv", index=False)
    top_fn.to_csv(out_dir / "final_top_false_negatives.csv", index=False)

    render_summary(
        out_dir / "final_results_summary.md",
        means,
        per_fold,
        paper,
        args.final_method,
        host_metrics,
        episode_metrics,
        contamination,
    )
    render_error_markdown(
        out_dir / "final_error_analysis.md",
        preds,
        host_metrics,
        episode_metrics,
        contamination,
        top_fp,
        top_fn,
    )
    final = means[means["method"] == args.final_method].iloc[0]
    print(out_dir / "final_results_summary.md")
    print(out_dir / "final_error_analysis.md")
    print(
        args.final_method,
        "auc",
        f"{float(final['auc_roc']):.4f}",
        "cal_ba",
        f"{float(final['calibrated_balanced_accuracy']):.4f}",
    )


if __name__ == "__main__":
    main()
