from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import binary_metrics, bootstrap_ci
from src.utils import ensure_dir, write_json


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
    merged = merged.rename(columns={"score_lie": "visual_score_lie"})
    merged["score_lie"] = alpha_visual * merged["visual_score_lie"] + (1.0 - alpha_visual) * merged["text_score_lie"]
    return merged[["video_id", "group_id", "label", "visual_score_lie", "text_score_lie", "score_lie"]]


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ensure_dir(args.out_dir)
    visual_val = average_predictions(args.visual_val)
    visual_test = average_predictions(args.visual_test)
    text_val = pd.read_csv(args.text_val)
    text_test = pd.read_csv(args.text_test)

    search_rows: list[dict[str, float]] = []
    best_alpha = 0.0
    best_auc = -1.0
    for alpha in np.linspace(0.0, 1.0, 21):
        fused_val = fuse_predictions(visual_val, text_val, float(alpha))
        metrics = binary_metrics(fused_val["label"].to_numpy(), fused_val["score_lie"].to_numpy())
        auc = float(metrics["auc_roc"])
        search_rows.append({"alpha_visual": float(alpha), "val_auc_roc": auc})
        if auc > best_auc:
            best_auc = auc
            best_alpha = float(alpha)

    fused_test = fuse_predictions(visual_test, text_test, best_alpha)
    test_metrics = binary_metrics(fused_test["label"].to_numpy(), fused_test["score_lie"].to_numpy())
    test_metrics["bootstrap_95ci"] = bootstrap_ci(
        fused_test["label"].to_numpy(),
        fused_test["score_lie"].to_numpy(),
        seed=args.seed,
    )

    pd.DataFrame(search_rows).to_csv(out_dir / "alpha_search.csv", index=False)
    fused_test.to_csv(out_dir / "test_predictions.csv", index=False)
    summary = {
        "best_alpha_visual": best_alpha,
        "best_val_auc_roc": best_auc,
        "test_metrics": test_metrics,
        "visual_val_predictions": [str(path) for path in args.visual_val],
        "visual_test_predictions": [str(path) for path in args.visual_test],
        "text_val_predictions": str(args.text_val),
        "text_test_predictions": str(args.text_test),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Late fusion for visual and text predictions.")
    parser.add_argument(
        "--visual-val",
        nargs="+",
        default=[
            "outputs/metrics/baseline_frame_resnet18/val_predictions.csv",
            "outputs/metrics/baseline_frame_resnet18_seed123/val_predictions.csv",
            "outputs/metrics/baseline_frame_resnet18_seed2025/val_predictions.csv",
        ],
    )
    parser.add_argument(
        "--visual-test",
        nargs="+",
        default=[
            "outputs/metrics/baseline_frame_resnet18/test_predictions.csv",
            "outputs/metrics/baseline_frame_resnet18_seed123/test_predictions.csv",
            "outputs/metrics/baseline_frame_resnet18_seed2025/test_predictions.csv",
        ],
    )
    parser.add_argument(
        "--text-val",
        default="outputs/metrics/real_life_text_baselines/tfidf_logistic_regression_val_predictions.csv",
    )
    parser.add_argument(
        "--text-test",
        default="outputs/metrics/real_life_text_baselines/tfidf_logistic_regression_test_predictions.csv",
    )
    parser.add_argument("--out-dir", default="outputs/metrics/real_life_late_fusion")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(summary)


if __name__ == "__main__":
    main()
