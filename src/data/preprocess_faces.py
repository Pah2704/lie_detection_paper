from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop faces from videos with MediaPipe.")
    parser.add_argument("--metadata", default="outputs/metrics/real_life_data_check/metadata.csv")
    parser.add_argument("--out-dir", default="data/processed/faces/real_life")
    parser.add_argument("--report-dir", default="outputs/metrics/real_life_face_check")
    parser.add_argument("--preview-dir", default="outputs/figures/real_life_face_previews")
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--model-selection", type=int, default=1, choices=[0, 1])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preview-count", type=int, default=24)
    return parser.parse_args()


def square_crop_from_bbox(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_size: int,
    margin: float,
) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * (1.0 + margin)
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - width)
    pad_bottom = max(0, y2 - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = center_crop(frame, output_size)
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def center_crop(frame: np.ndarray, output_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    x1 = (width - side) // 2
    y1 = (height - side) // 2
    crop = frame[y1 : y1 + side, x1 : x1 + side]
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def detection_to_bbox(detection: Any, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
    rel = detection.location_data.relative_bounding_box
    x = int(round(rel.xmin * frame_width))
    y = int(round(rel.ymin * frame_height))
    w = int(round(rel.width * frame_width))
    h = int(round(rel.height * frame_height))
    x = max(0, min(x, frame_width - 1))
    y = max(0, min(y, frame_height - 1))
    w = max(1, min(w, frame_width - x))
    h = max(1, min(h, frame_height - y))
    return x, y, w, h


def choose_largest_detection(detections: list[Any], frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
    if not detections:
        return None
    boxes = [detection_to_bbox(det, frame_width, frame_height) for det in detections]
    return max(boxes, key=lambda box: box[2] * box[3])


def output_path_for(row: pd.Series, out_dir: Path) -> Path:
    stem = Path(str(row["video_id"])).stem
    return out_dir / f"{stem}.mp4"


def process_video(
    row: pd.Series,
    detector: Any,
    out_dir: Path,
    output_size: int,
    margin: float,
    detect_every: int,
    overwrite: bool,
) -> tuple[dict[str, Any], np.ndarray | None]:
    video_path = Path(str(row["video_path"]))
    out_path = output_path_for(row, out_dir)
    ensure_dir(out_path.parent)

    if out_path.exists() and not overwrite:
        existing = {
            "video_id": row["video_id"],
            "label": row.get("label"),
            "group_id": row.get("group_id"),
            "source_video_path": str(video_path),
            "face_video_path": str(out_path),
            "status": "exists",
        }
        return existing, None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "video_id": row["video_id"],
            "label": row.get("label"),
            "group_id": row.get("group_id"),
            "source_video_path": str(video_path),
            "face_video_path": str(out_path),
            "status": "open_failed",
        }, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_size, output_size),
    )
    if not writer.isOpened():
        cap.release()
        return {
            "video_id": row["video_id"],
            "label": row.get("label"),
            "group_id": row.get("group_id"),
            "source_video_path": str(video_path),
            "face_video_path": str(out_path),
            "status": "writer_failed",
        }, None

    total_frames = 0
    detected_frames = 0
    reused_frames = 0
    center_frames = 0
    detect_attempts = 0
    last_bbox: tuple[int, int, int, int] | None = None
    first_crop: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx = total_frames
        total_frames += 1
        frame_height, frame_width = frame.shape[:2]
        should_detect = last_bbox is None or frame_idx % max(detect_every, 1) == 0
        bbox = None
        if should_detect:
            detect_attempts += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            bbox = choose_largest_detection(result.detections or [], frame_width, frame_height)
            if bbox is not None:
                last_bbox = bbox
                detected_frames += 1

        if bbox is None and last_bbox is not None:
            bbox = last_bbox
            reused_frames += 1

        if bbox is not None:
            crop = square_crop_from_bbox(frame, bbox, output_size, margin)
        else:
            crop = center_crop(frame, output_size)
            center_frames += 1

        if first_crop is None:
            first_crop = crop.copy()
        writer.write(crop)

    cap.release()
    writer.release()

    usable_frames = total_frames - center_frames
    metadata = {
        "video_id": row["video_id"],
        "label": row.get("label"),
        "group_id": row.get("group_id"),
        "source_video_path": str(video_path),
        "face_video_path": str(out_path),
        "status": "ok" if total_frames > 0 else "empty",
        "total_frames": total_frames,
        "detect_attempts": detect_attempts,
        "detected_frames": detected_frames,
        "reused_frames": reused_frames,
        "center_frames": center_frames,
        "face_crop_rate": usable_frames / total_frames if total_frames else 0.0,
        "new_detection_rate": detected_frames / max(detect_attempts, 1),
        "fps": fps,
        "output_size": output_size,
        "detect_every": detect_every,
        "margin": margin,
    }
    return metadata, first_crop


def write_contact_sheet(previews: list[tuple[str, np.ndarray]], path: Path, cell_size: int = 128) -> None:
    if not previews:
        return
    cols = 6
    rows = int(np.ceil(len(previews) / cols))
    label_height = 22
    sheet = np.full((rows * (cell_size + label_height), cols * cell_size, 3), 255, dtype=np.uint8)
    for idx, (label, image) in enumerate(previews):
        row = idx // cols
        col = idx % cols
        y = row * (cell_size + label_height)
        x = col * cell_size
        image = cv2.resize(image, (cell_size, cell_size), interpolation=cv2.INTER_AREA)
        sheet[y : y + cell_size, x : x + cell_size] = image
        cv2.putText(
            sheet,
            label[:18],
            (x + 4, y + cell_size + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    ensure_dir(path.parent)
    cv2.imwrite(str(path), sheet)


def main() -> None:
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    if args.limit:
        metadata = metadata.head(args.limit)

    out_dir = ensure_dir(args.out_dir)
    report_dir = ensure_dir(args.report_dir)
    preview_dir = ensure_dir(args.preview_dir)

    rows: list[dict[str, Any]] = []
    previews: list[tuple[str, np.ndarray]] = []
    mp_face_detection = mp.solutions.face_detection

    with mp_face_detection.FaceDetection(
        model_selection=args.model_selection,
        min_detection_confidence=args.min_confidence,
    ) as detector:
        for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Cropping faces"):
            result, preview = process_video(
                row=row,
                detector=detector,
                out_dir=out_dir,
                output_size=args.output_size,
                margin=args.margin,
                detect_every=args.detect_every,
                overwrite=args.overwrite,
            )
            rows.append(result)
            if preview is not None and len(previews) < args.preview_count:
                previews.append((Path(str(row["video_id"])).stem, preview))

    face_df = pd.DataFrame(rows)
    face_df.to_csv(report_dir / "face_metadata.csv", index=False)
    if "face_crop_rate" in face_df:
        low = face_df[(face_df["status"] == "ok") & (face_df["face_crop_rate"] < 0.70)]
        low.to_csv(report_dir / "low_face_crop_rate.csv", index=False)
    write_contact_sheet(previews, preview_dir / "contact_sheet.jpg")

    summary: dict[str, Any] = {
        "num_videos": int(len(face_df)),
        "status_counts": face_df["status"].value_counts(dropna=False).to_dict() if "status" in face_df else {},
    }
    if "face_crop_rate" in face_df:
        ok = face_df[face_df["status"] == "ok"].copy()
        summary.update(
            {
                "mean_face_crop_rate": float(ok["face_crop_rate"].mean()) if not ok.empty else 0.0,
                "median_face_crop_rate": float(ok["face_crop_rate"].median()) if not ok.empty else 0.0,
                "videos_below_70pct": int((ok["face_crop_rate"] < 0.70).sum()) if not ok.empty else 0,
                "mean_new_detection_rate": float(ok["new_detection_rate"].mean()) if not ok.empty else 0.0,
            }
        )
    write_json(report_dir / "face_summary.json", summary)
    print(f"Wrote face crops to {out_dir}")
    print(f"Wrote face report to {report_dir}")
    print(summary)


if __name__ == "__main__":
    main()
