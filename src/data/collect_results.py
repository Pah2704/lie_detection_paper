from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import ensure_dir


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metric(metrics: dict[str, Any], key: str) -> float:
    return float(metrics.get(key, float("nan")))


def fmt(value: float, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def sort_value(value: Any) -> float:
    if isinstance(value, str):
        try:
            return float(value.split()[0])
        except (ValueError, IndexError):
            return float("nan")
    return float(value)


def add_row(
    rows: list[dict[str, Any]],
    model: str,
    modality: str,
    split_protocol: str,
    test_metrics: dict[str, Any],
    val_metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> None:
    rows.append(
        {
            "model": model,
            "modality": modality,
            "split_protocol": split_protocol,
            "val_auc_roc": metric(val_metrics or {}, "auc_roc"),
            "test_auc_roc": metric(test_metrics, "auc_roc"),
            "test_auc_pr": metric(test_metrics, "auc_pr"),
            "test_macro_f1": metric(test_metrics, "macro_f1"),
            "test_lie_recall": metric(test_metrics, "recall_lie"),
            "test_eer": metric(test_metrics, "eer"),
            "notes": notes,
        }
    )


def markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    metric_cols = [
        "val_auc_roc",
        "test_auc_roc",
        "test_auc_pr",
        "test_macro_f1",
        "test_lie_recall",
        "test_eer",
    ]
    for col in metric_cols:
        display[col] = display[col].map(fmt)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines) + "\n"


def build_results(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    baselines = read_json(args.non_learning)
    baseline_names = {
        "majority": "Majority baseline",
        "random_stratified": "Random stratified",
        "metadata_logreg": "Metadata Logistic Regression",
        "metadata_svm": "Metadata SVM",
    }
    baseline_modality = {
        "majority": "Label only",
        "random_stratified": "Label only",
        "metadata_logreg": "Metadata",
        "metadata_svm": "Metadata",
    }
    for key, name in baseline_names.items():
        add_row(
            rows,
            name,
            baseline_modality[key],
            "group-disjoint",
            baselines["test"][key],
            baselines["val"][key],
            "Random baseline is one seed only" if key == "random_stratified" else "",
        )

    multiseed = read_json(args.frame_multiseed)
    agg = multiseed["aggregate"]
    rows.append(
        {
            "model": "ResNet18 frame",
            "modality": "Visual",
            "split_protocol": "group-disjoint, 3 seeds",
            "val_auc_roc": fmt_mean_std(agg["best_val_auc_roc"]["mean"], agg["best_val_auc_roc"]["std"]),
            "test_auc_roc": fmt_mean_std(agg["test_auc_roc"]["mean"], agg["test_auc_roc"]["std"]),
            "test_auc_pr": fmt_mean_std(agg["test_auc_pr"]["mean"], agg["test_auc_pr"]["std"]),
            "test_macro_f1": fmt_mean_std(agg["test_macro_f1"]["mean"], agg["test_macro_f1"]["std"]),
            "test_lie_recall": fmt_mean_std(agg["test_lie_recall"]["mean"], agg["test_lie_recall"]["std"]),
            "test_eer": fmt_mean_std(agg["test_eer"]["mean"], agg["test_eer"]["std"]),
            "notes": "Mean +/- std over seeds 42, 123, 2025",
        }
    )

    r3d = read_json(args.r3d_summary)
    r3d_val = read_json(args.r3d_val_metrics)
    add_row(rows, "R3D-18 RGB", "Visual", "group-disjoint", r3d["test_metrics"], r3d_val, "Negative result")

    text = read_json(args.text_baselines)
    add_row(
        rows,
        "TF-IDF Logistic Regression",
        "Text",
        "group-disjoint",
        text["tfidf_logistic_regression"]["test"],
        text["tfidf_logistic_regression"]["val"],
        "Transcript may contain trial/source shortcuts",
    )
    add_row(
        rows,
        "TF-IDF Linear SVM",
        "Text",
        "group-disjoint",
        text["tfidf_linear_svm"]["test"],
        text["tfidf_linear_svm"]["val"],
        "Default threshold has low Lie recall",
    )

    fusion = read_json(args.fusion_summary)
    add_row(
        rows,
        "Late fusion",
        "Visual ensemble + Text LR",
        "group-disjoint",
        fusion["test_metrics"],
        {"auc_roc": fusion["best_val_auc_roc"]},
        f"alpha_visual={fusion['best_alpha_visual']:.2f}",
    )

    return pd.DataFrame(rows)


def write_markdown(df: pd.DataFrame, out_path: Path) -> None:
    main = df[~df["model"].isin(["Majority baseline", "Random stratified"])].copy()
    sanity = df[df["model"].isin(["Majority baseline", "Random stratified"])].copy()
    main["_rank_auc"] = main["test_auc_roc"].map(sort_value)
    main = main.sort_values("_rank_auc", ascending=False).drop(columns=["_rank_auc"])
    sanity["_rank_auc"] = sanity["test_auc_roc"].map(sort_value)
    sanity = sanity.sort_values("_rank_auc", ascending=False).drop(columns=["_rank_auc"])
    text = "# Final Results\n\n"
    text += "## Main Models\n\n"
    text += markdown_table(main)
    text += "\n## Sanity Baselines\n\n"
    text += markdown_table(sanity)
    text += "\nNotes:\n\n"
    text += "- Main model ranking uses test AUC-ROC and excludes label-only sanity baselines.\n"
    text += "- ResNet18 frame is reported as mean +/- std over 3 seeds.\n"
    text += "- Random stratified is retained as a sanity baseline, not as a meaningful model; its high score reflects small-test-set variance from one random seed.\n"
    text += "- Text and fusion results may include trial/source-content shortcuts and should be interpreted separately from behavioral visual models.\n"
    out_path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect all experiment results into final report tables.")
    parser.add_argument("--non-learning", default="outputs/metrics/real_life_baselines/non_learning_baselines.json")
    parser.add_argument("--frame-multiseed", default="outputs/metrics/baseline_frame_resnet18_multiseed/summary.json")
    parser.add_argument("--r3d-summary", default="outputs/metrics/rgb_r3d18/summary.json")
    parser.add_argument("--r3d-val-metrics", default="outputs/metrics/rgb_r3d18/val_metrics.json")
    parser.add_argument("--text-baselines", default="outputs/metrics/real_life_text_baselines/text_baselines.json")
    parser.add_argument("--fusion-summary", default="outputs/metrics/real_life_late_fusion/summary.json")
    parser.add_argument("--out-dir", default="outputs/metrics/final_report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    df = build_results(args)
    csv_path = out_dir / "final_results_table.csv"
    md_path = out_dir / "final_results_summary.md"
    df.to_csv(csv_path, index=False)
    write_markdown(df, md_path)
    print(df.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
