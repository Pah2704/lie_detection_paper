from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils import log_args, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select examples for Grad-CAM/attention rollout reporting.")
    parser.add_argument("--predictions", required=True, help="Clip-level prediction CSV.")
    parser.add_argument("--out", default="outputs/metrics/visualization_candidates.csv")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--log-dir", default="outputs/logs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("visualize_results", args.log_dir)
    logger.info("Log file: %s", log_path)
    log_args(logger, args)
    df = pd.read_csv(args.predictions)
    if not {"label", "score_lie"}.issubset(df.columns):
        raise ValueError("predictions must contain label and score_lie columns")
    df = df.copy()
    df["pred"] = (df["score_lie"] >= 0.5).astype(int)
    df["confidence"] = (df["score_lie"] - 0.5).abs()
    df["correct"] = df["pred"] == df["label"].astype(int)
    candidates = pd.concat(
        [
            df[df["correct"]].sort_values("confidence", ascending=False).head(args.top_k),
            df[~df["correct"]].sort_values("confidence", ascending=False).head(args.top_k),
        ],
        ignore_index=True,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out, index=False)
    logger.info("Wrote %d visualization candidates to %s", len(candidates), out)


if __name__ == "__main__":
    main()
