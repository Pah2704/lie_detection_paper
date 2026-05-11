from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.utils import ensure_dir


def read_middle_frame(video_path: str | Path, size: int = 160) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((size, size, 3), dtype=np.uint8)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2, 0))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return np.zeros((size, size, 3), dtype=np.uint8)
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return frame


def choose_examples(report: pd.DataFrame, per_type: int) -> pd.DataFrame:
    chosen = []
    for error_type in ["TP", "TN", "FP", "FN"]:
        subset = report[report["error_type"] == error_type].copy()
        if subset.empty:
            continue
        if error_type in {"TP", "TN"}:
            subset = subset.sort_values("abs_margin_from_threshold", ascending=False)
        else:
            subset = subset.sort_values("abs_margin_from_threshold", ascending=True)
        chosen.append(subset.head(per_type))
    if not chosen:
        return pd.DataFrame()
    return pd.concat(chosen, ignore_index=True)


def label_lines(row: pd.Series, score_col: str) -> list[str]:
    label_name = "lie" if int(row["label"]) == 1 else "truth"
    pred_name = "lie" if int(row["pred_label"]) == 1 else "truth"
    return [
        str(row["video_id"]),
        f"{row['error_type']} y={label_name} p={pred_name}",
        f"score={float(row[score_col]):.3f}",
        str(row.get("group_id", ""))[:28],
    ]


def draw_tile(frame: np.ndarray, lines: list[str], width: int, height: int) -> np.ndarray:
    tile = np.full((height, width, 3), 255, dtype=np.uint8)
    tile[: frame.shape[0], : frame.shape[1]] = frame
    y = frame.shape[0] + 22
    for line in lines:
        cv2.putText(tile, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        y += 20
    return tile


def make_sheet(examples: pd.DataFrame, score_col: str, out_path: Path, tile_size: int = 160, cols: int = 4) -> None:
    tile_w = tile_size
    tile_h = tile_size + 92
    tiles = []
    for _, row in examples.iterrows():
        frame = read_middle_frame(row["face_video_path"], size=tile_size)
        tiles.append(draw_tile(frame, label_lines(row, score_col), tile_w, tile_h))
    if not tiles:
        return
    rows = int(np.ceil(len(tiles) / cols))
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 240, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        sheet[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = tile
    cv2.imwrite(str(out_path), sheet)


def write_markdown(examples: pd.DataFrame, score_col: str, image_path: Path, out_path: Path) -> None:
    rel_image = Path("../figures/report_examples") / image_path.name
    lines = [
        "# Visual Prediction Examples",
        "",
        f"![Prediction examples]({rel_image.as_posix()})",
        "",
        "| Type | Video | Label | Prediction | Score | Group | Notes |",
        "|---|---|---|---|---:|---|---|",
    ]
    for _, row in examples.iterrows():
        label_name = "deceptive" if int(row["label"]) == 1 else "truthful"
        pred_name = "deceptive" if int(row["pred_label"]) == 1 else "truthful"
        note = "near threshold" if float(row["abs_margin_from_threshold"]) < 0.06 else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["error_type"]),
                    str(row["video_id"]),
                    label_name,
                    pred_name,
                    f"{float(row[score_col]):.3f}",
                    str(row.get("group_id", "")),
                    note,
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ensure_dir(args.out_dir)
    report_out = ensure_dir(args.report_out_dir)
    report = pd.read_csv(args.error_report)
    score_cols = [col for col in report.columns if col.endswith("_score_lie")]
    if not score_cols:
        raise ValueError("No score column ending with _score_lie found")
    score_col = args.score_column or score_cols[-1]
    examples = choose_examples(report, args.per_type)
    examples_path = out_dir / "visual_examples.csv"
    examples.to_csv(examples_path, index=False)
    image_path = out_dir / "visual_examples_contact_sheet.jpg"
    make_sheet(examples, score_col, image_path, args.tile_size, args.cols)
    markdown_path = report_out / "visual_examples.md"
    write_markdown(examples, score_col, image_path, markdown_path)
    return {
        "examples": str(examples_path),
        "image": str(image_path),
        "markdown": str(markdown_path),
        "n_examples": int(len(examples)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create visual examples contact sheet from error analysis.")
    parser.add_argument("--error-report", default="outputs/metrics/error_analysis/late_fusion_error_report.csv")
    parser.add_argument("--score-column", default="")
    parser.add_argument("--out-dir", default="outputs/figures/report_examples")
    parser.add_argument("--report-out-dir", default="outputs/metrics/final_report")
    parser.add_argument("--per-type", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=160)
    parser.add_argument("--cols", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
