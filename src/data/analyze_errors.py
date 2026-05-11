from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import binary_metrics
from src.utils import ensure_dir, write_json


KEEP_METADATA_COLUMNS = [
    "video_id",
    "class",
    "role",
    "trial_name",
    "ruling",
    "source_link",
    "fps",
    "num_frames",
    "duration_sec",
    "width",
    "height",
    "file_size_bytes",
]

KEEP_FACE_COLUMNS = [
    "video_id",
    "face_crop_rate",
    "new_detection_rate",
    "center_frames",
    "reused_frames",
    "total_frames",
    "face_video_path",
]


def label_error_type(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 0:
        return "TN"
    if label == 0 and pred == 1:
        return "FP"
    return "FN"


def load_predictions(path: str | Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "score_lie" not in df.columns:
        raise ValueError(f"{path} must contain score_lie")
    return df.rename(columns={"score_lie": f"{model_name}_score_lie"})


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ensure_dir(args.out_dir)
    pred = load_predictions(args.predictions, args.model_name)
    score_col = f"{args.model_name}_score_lie"

    metadata = pd.read_csv(args.metadata)
    face = pd.read_csv(args.face_metadata)

    available_metadata = [col for col in KEEP_METADATA_COLUMNS if col in metadata.columns]
    available_face = [col for col in KEEP_FACE_COLUMNS if col in face.columns]

    report = pred.merge(metadata[available_metadata], on="video_id", how="left")
    report = report.merge(face[available_face], on="video_id", how="left", suffixes=("", "_face"))
    report["pred_label"] = (report[score_col] >= args.threshold).astype(int)
    report["error_type"] = [
        label_error_type(int(label), int(pred_label))
        for label, pred_label in zip(report["label"], report["pred_label"])
    ]
    report["abs_margin_from_threshold"] = (report[score_col] - args.threshold).abs()

    sort_cols = ["error_type", "abs_margin_from_threshold"]
    report = report.sort_values(sort_cols, ascending=[True, True])
    report.to_csv(out_dir / f"{args.model_name}_error_report.csv", index=False)

    errors = report[report["error_type"].isin(["FP", "FN"])].copy()
    errors.to_csv(out_dir / f"{args.model_name}_errors_only.csv", index=False)

    y_true = report["label"].to_numpy()
    y_score = report[score_col].to_numpy()
    metrics = binary_metrics(y_true, y_score, threshold=args.threshold)

    summary: dict[str, Any] = {
        "model_name": args.model_name,
        "prediction_file": str(args.predictions),
        "threshold": args.threshold,
        "metrics": metrics,
        "counts_by_error_type": report["error_type"].value_counts().to_dict(),
        "errors": int(len(errors)),
        "n_videos": int(len(report)),
    }

    for col in ["group_id", "role", "trial_name"]:
        if col in report.columns:
            summary[f"errors_by_{col}"] = (
                errors.groupby(col, dropna=False).size().sort_values(ascending=False).head(10).to_dict()
            )

    quality_cols = ["face_crop_rate", "new_detection_rate", "center_frames", "duration_sec", "fps", "width", "height"]
    quality_summary: dict[str, Any] = {}
    for col in quality_cols:
        if col in report.columns:
            quality_summary[col] = {
                "correct_mean": float(report.loc[report["error_type"].isin(["TP", "TN"]), col].mean()),
                "error_mean": float(errors[col].mean()) if not errors.empty else float("nan"),
            }
    summary["quality_summary"] = quality_summary

    write_json(out_dir / f"{args.model_name}_error_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze video-level prediction errors.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--metadata", default="outputs/metrics/real_life_data_check/metadata.csv")
    parser.add_argument("--face-metadata", default="outputs/metrics/real_life_face_check/face_metadata.csv")
    parser.add_argument("--out-dir", default="outputs/metrics/error_analysis")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_report(args)
    print(summary)


if __name__ == "__main__":
    main()
