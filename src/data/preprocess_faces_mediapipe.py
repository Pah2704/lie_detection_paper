from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import ensure_dir, log_args, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract aligned/masked 224x224 face frames with MediaPipe Face Mesh.")
    parser.add_argument("--metadata", required=True, help="CSV with video_id/video_path/label.")
    parser.add_argument("--dataset", default="dolos", choices=["dolos"])
    parser.add_argument("--out-root", default="data/processed/faces_224_clean")
    parser.add_argument("--manifest-root", default="data/processed/faces_224_clean")
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--max-num-faces", type=int, default=4)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.35)
    parser.add_argument("--iou-reinit-threshold", type=float, default=0.3)
    parser.add_argument("--track-selection", choices=["online_iou", "dominant_after_start"], default="dominant_after_start")
    parser.add_argument("--track-selection-start-seconds", type=float, default=1.0)
    parser.add_argument("--max-track-gap-frames", type=int, default=12)
    parser.add_argument("--min-selected-track-ratio", type=float, default=0.45)
    parser.add_argument("--min-selected-track-frames", type=int, default=8)
    parser.add_argument("--max-fail-rate", type=float, default=0.4)
    parser.add_argument("--max-output-frames", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-dir", default="outputs/logs")
    return parser.parse_args()


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def landmarks_to_points(face_landmarks: Any, width: int, height: int) -> np.ndarray:
    points = np.array(
        [[lm.x * width, lm.y * height] for lm in face_landmarks.landmark],
        dtype=np.float32,
    )
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def points_to_bbox(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    x1 = int(np.clip(np.floor(x1), 0, width - 1))
    y1 = int(np.clip(np.floor(y1), 0, height - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, width))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, height))
    return x1, y1, x2 - x1, y2 - y1


def center_distance_score(bbox: tuple[int, int, int, int], width: int, height: int) -> float:
    x, y, w, h = bbox
    cx = x + w / 2
    cy = y + h / 2
    return -float((cx - width / 2) ** 2 + (cy - height / 2) ** 2)


def choose_initial_face(candidates: list[dict[str, Any]], width: int, height: int) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c["bbox"][2] * c["bbox"][3], center_distance_score(c["bbox"], width, height)))


def choose_tracked_face(
    candidates: list[dict[str, Any]],
    previous_bbox: tuple[int, int, int, int] | None,
    width: int,
    height: int,
    iou_threshold: float,
) -> tuple[dict[str, Any] | None, bool]:
    if not candidates:
        return None, False
    if previous_bbox is None:
        return choose_initial_face(candidates, width, height), True
    best = max(candidates, key=lambda c: bbox_iou(previous_bbox, c["bbox"]))
    if bbox_iou(previous_bbox, best["bbox"]) >= iou_threshold:
        return best, False
    return choose_initial_face(candidates, width, height), True


def mask_background(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    hull = cv2.convexHull(points.astype(np.int32))
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def square_crop(frame: np.ndarray, bbox: tuple[int, int, int, int], output_size: int, margin: float) -> np.ndarray:
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
        frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        x1 += pad_left
        x2 += pad_left
        y1 += pad_top
        y2 += pad_top
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def detect_candidates(face_mesh: Any, frame: np.ndarray) -> list[dict[str, Any]]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return []
    height, width = frame.shape[:2]
    candidates: list[dict[str, Any]] = []
    for landmarks in result.multi_face_landmarks:
        points = landmarks_to_points(landmarks, width, height)
        candidates.append({"points": points, "bbox": points_to_bbox(points, width, height)})
    return candidates


def candidate_area(candidate: dict[str, Any]) -> int:
    _, _, width, height = candidate["bbox"]
    return int(width * height)


def assign_track_ids(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]] = []
    next_track_id = 0
    for sample in samples:
        used_tracks: set[int] = set()
        candidates = sorted(sample["candidates"], key=candidate_area, reverse=True)
        for candidate in candidates:
            best_track: dict[str, Any] | None = None
            best_iou = 0.0
            for track in tracks:
                if int(track["track_id"]) in used_tracks:
                    continue
                gap = int(sample["frame_index"]) - int(track["last_frame_index"])
                if gap > args.max_track_gap_frames:
                    continue
                iou = bbox_iou(track["last_bbox"], candidate["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            if best_track is None or best_iou < args.iou_reinit_threshold:
                best_track = {
                    "track_id": next_track_id,
                    "detections": [],
                    "last_bbox": candidate["bbox"],
                    "last_frame_index": sample["frame_index"],
                }
                tracks.append(best_track)
                next_track_id += 1
            candidate["track_id"] = int(best_track["track_id"])
            candidate["track_iou"] = float(best_iou)
            detection = {
                "frame_index": int(sample["frame_index"]),
                "timestamp": float(sample["timestamp"]),
                "bbox": candidate["bbox"],
                "area": candidate_area(candidate),
            }
            best_track["detections"].append(detection)
            best_track["last_bbox"] = candidate["bbox"]
            best_track["last_frame_index"] = int(sample["frame_index"])
            used_tracks.add(int(best_track["track_id"]))
    return tracks


def choose_dominant_track(samples: list[dict[str, Any]], tracks: list[dict[str, Any]], args: argparse.Namespace) -> tuple[int | None, bool, dict[str, Any]]:
    if not tracks:
        return None, False, {
            "selected_track_id": None,
            "selected_track_frames": 0,
            "selected_track_frames_after_start": 0,
            "selected_track_after_start_ratio": 0.0,
        }
    start_seconds = max(0.0, float(args.track_selection_start_seconds))
    after_start_frames = [sample for sample in samples if float(sample["timestamp"]) >= start_seconds]
    if not after_start_frames:
        after_start_frames = samples
        start_seconds = 0.0
    after_start_count = max(len(after_start_frames), 1)

    scored: list[tuple[tuple[int, int, float], dict[str, Any], list[dict[str, Any]]]] = []
    for track in tracks:
        detections = track["detections"]
        after = [det for det in detections if float(det["timestamp"]) >= start_seconds]
        mean_area = float(np.mean([det["area"] for det in after or detections])) if detections else 0.0
        scored.append(((len(after), len(detections), mean_area), track, after))
    _, selected, selected_after = max(scored, key=lambda item: item[0])
    selected_after_count = len(selected_after)
    selected_ratio = selected_after_count / after_start_count
    min_track_frames = min(int(args.min_selected_track_frames), after_start_count)
    clip_face_valid = selected_after_count >= min_track_frames and selected_ratio >= float(args.min_selected_track_ratio)
    stats = {
        "selected_track_id": int(selected["track_id"]),
        "selected_track_frames": int(len(selected["detections"])),
        "selected_track_frames_after_start": int(selected_after_count),
        "selected_track_after_start_ratio": float(selected_ratio),
        "track_selection_start_seconds": float(start_seconds),
        "num_tracks": int(len(tracks)),
        "clip_face_valid": bool(clip_face_valid),
    }
    return int(selected["track_id"]), bool(clip_face_valid), stats


def write_face_validity(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(rows).to_csv(path, index=False)


def process_video_online_iou(
    row: pd.Series,
    face_mesh: Any,
    out_dir: Path,
    face_valid_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_id = str(row["video_id"])
    video_path = Path(str(row["video_path"]))
    video_out_dir = out_dir / video_id
    face_valid_path = face_valid_dir / f"{video_id}.csv"
    if video_out_dir.exists() and any(video_out_dir.glob("*.jpg")) and not args.overwrite:
        frame_count = len(list(video_out_dir.glob("*.jpg")))
        status = "exists" if frame_count <= args.max_output_frames else "too_many_frames"
        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "frames_dir": str(video_out_dir),
            "status": status,
            "output_frames": frame_count,
            "face_valid_path": str(face_valid_path) if face_valid_path.exists() else "",
        }
    ensure_dir(video_out_dir)
    if args.overwrite:
        for old in video_out_dir.glob("*.jpg"):
            old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video_id": video_id, "video_path": str(video_path), "frames_dir": str(video_out_dir), "status": "open_failed"}

    source_fps = cap.get(cv2.CAP_PROP_FPS) or args.target_fps
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    next_time = 0.0
    frame_idx = 0
    out_idx = 0
    fail_frames = 0
    reused_frames = 0
    detected_frames = 0
    multi_face_frames = 0
    reinit_count = 0
    last_bbox: tuple[int, int, int, int] | None = None
    last_crop: np.ndarray | None = None
    validity_rows: list[dict[str, Any]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / max(source_fps, 1e-6)
        frame_idx += 1
        if timestamp + 1e-9 < next_time:
            continue
        next_time += 1.0 / args.target_fps

        height, width = frame.shape[:2]
        candidates = detect_candidates(face_mesh, frame)
        if len(candidates) > 1:
            multi_face_frames += 1
        target, reinit = choose_tracked_face(candidates, last_bbox, width, height, args.iou_reinit_threshold)
        if target is None:
            fail_frames += 1
            if last_crop is not None:
                crop = last_crop.copy()
                reused_frames += 1
                reused = True
            else:
                crop = np.zeros((args.output_size, args.output_size, 3), dtype=np.uint8)
                reused = False
            face_valid = False
            track_id = -1
        else:
            detected_frames += 1
            reinit_count += int(reinit)
            last_bbox = target["bbox"]
            masked = mask_background(frame, target["points"])
            crop = square_crop(masked, target["bbox"], args.output_size, args.margin)
            last_crop = crop.copy()
            reused = False
            face_valid = True
            track_id = 0
        cv2.imwrite(str(video_out_dir / f"{out_idx:06d}.jpg"), crop)
        validity_rows.append(
            {
                "frame_index": int(out_idx),
                "timestamp": float(timestamp),
                "source_frame_index": int(frame_idx - 1),
                "face_valid": int(face_valid),
                "face_valid_ratio": float(face_valid),
                "track_id": int(track_id),
                "num_faces": int(len(candidates)),
                "reused_previous_crop": int(reused),
                "track_selection": "online_iou",
            }
        )
        out_idx += 1
        if out_idx > args.max_output_frames:
            break

    cap.release()
    write_face_validity(face_valid_path, validity_rows)
    fail_rate = fail_frames / max(out_idx, 1)
    if out_idx > args.max_output_frames:
        status = "too_many_frames"
    else:
        status = "ok" if out_idx > 0 and fail_rate <= args.max_fail_rate else "too_many_failures"
    return {
        "video_id": video_id,
        "label": row.get("label"),
        "group_id": row.get("group_id"),
        "video_path": str(video_path),
        "frames_dir": str(video_out_dir),
        "status": status,
        "source_fps": float(source_fps),
        "target_fps": float(args.target_fps),
        "source_frames": int(source_frames),
        "output_frames": int(out_idx),
        "detected_frames": int(detected_frames),
        "reused_frames": int(reused_frames),
        "fail_frames": int(fail_frames),
        "fail_rate": float(fail_rate),
        "multi_face_frames": int(multi_face_frames),
        "reinit_count": int(reinit_count),
        "clip_face_valid": bool(fail_rate <= args.max_fail_rate),
        "face_valid_path": str(face_valid_path),
        "output_size": int(args.output_size),
    }


def process_video_dominant_track(
    row: pd.Series,
    face_mesh: Any,
    out_dir: Path,
    face_valid_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    video_id = str(row["video_id"])
    video_path = Path(str(row["video_path"]))
    video_out_dir = out_dir / video_id
    face_valid_path = face_valid_dir / f"{video_id}.csv"
    if video_out_dir.exists() and any(video_out_dir.glob("*.jpg")) and not args.overwrite:
        frame_count = len(list(video_out_dir.glob("*.jpg")))
        status = "exists" if frame_count <= args.max_output_frames else "too_many_frames"
        return {
            "video_id": video_id,
            "video_path": str(video_path),
            "frames_dir": str(video_out_dir),
            "status": status,
            "output_frames": frame_count,
            "face_valid_path": str(face_valid_path) if face_valid_path.exists() else "",
        }
    ensure_dir(video_out_dir)
    if args.overwrite:
        for old in video_out_dir.glob("*.jpg"):
            old.unlink()
        if face_valid_path.exists():
            face_valid_path.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video_id": video_id, "video_path": str(video_path), "frames_dir": str(video_out_dir), "status": "open_failed"}

    source_fps = cap.get(cv2.CAP_PROP_FPS) or args.target_fps
    source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    next_time = 0.0
    frame_idx = 0
    out_idx = 0
    samples: list[dict[str, Any]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / max(source_fps, 1e-6)
        source_frame_index = frame_idx
        frame_idx += 1
        if timestamp + 1e-9 < next_time:
            continue
        next_time += 1.0 / args.target_fps
        candidates = detect_candidates(face_mesh, frame)
        samples.append(
            {
                "frame_index": int(out_idx),
                "source_frame_index": int(source_frame_index),
                "timestamp": float(timestamp),
                "candidates": candidates,
            }
        )
        out_idx += 1
        if out_idx > args.max_output_frames:
            break
    cap.release()

    if not samples:
        return {
            "video_id": video_id,
            "label": row.get("label"),
            "group_id": row.get("group_id"),
            "video_path": str(video_path),
            "frames_dir": str(video_out_dir),
            "status": "no_frames",
            "source_fps": float(source_fps),
            "target_fps": float(args.target_fps),
            "source_frames": int(source_frames),
            "output_frames": 0,
        }

    multi_face_frames = sum(1 for sample in samples if len(sample["candidates"]) > 1)
    tracks = assign_track_ids(samples, args)
    selected_track_id, clip_face_valid, track_stats = choose_dominant_track(samples, tracks, args)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video_id": video_id, "video_path": str(video_path), "frames_dir": str(video_out_dir), "status": "open_failed_second_pass"}

    samples_by_source = {int(sample["source_frame_index"]): sample for sample in samples}
    last_crop: np.ndarray | None = None
    validity_rows: list[dict[str, Any]] = []
    detected_frames = 0
    reused_frames = 0
    fail_frames = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sample = samples_by_source.get(frame_idx)
        frame_idx += 1
        if sample is None:
            continue
        target = None
        if selected_track_id is not None:
            target = next((candidate for candidate in sample["candidates"] if candidate.get("track_id") == selected_track_id), None)
        if target is not None and clip_face_valid:
            masked = mask_background(frame, target["points"])
            crop = square_crop(masked, target["bbox"], args.output_size, args.margin)
            last_crop = crop.copy()
            face_valid = True
            reused = False
            detected_frames += 1
        else:
            fail_frames += 1
            face_valid = False
            if last_crop is not None:
                crop = last_crop.copy()
                reused = True
                reused_frames += 1
            else:
                crop = np.zeros((args.output_size, args.output_size, 3), dtype=np.uint8)
                reused = False
        out_frame_index = int(sample["frame_index"])
        cv2.imwrite(str(video_out_dir / f"{out_frame_index:06d}.jpg"), crop)
        validity_rows.append(
            {
                "frame_index": out_frame_index,
                "timestamp": float(sample["timestamp"]),
                "source_frame_index": int(sample["source_frame_index"]),
                "face_valid": int(face_valid),
                "face_valid_ratio": float(face_valid),
                "track_id": int(selected_track_id) if selected_track_id is not None else -1,
                "num_faces": int(len(sample["candidates"])),
                "reused_previous_crop": int(reused),
                "clip_face_valid": int(clip_face_valid),
                "selected_track_id": int(selected_track_id) if selected_track_id is not None else -1,
                "track_selection": "dominant_after_start",
            }
        )
    cap.release()
    write_face_validity(face_valid_path, validity_rows)

    out_frames = len(samples)
    fail_rate = fail_frames / max(out_frames, 1)
    status = "too_many_frames" if out_frames > args.max_output_frames else "ok"
    return {
        "video_id": video_id,
        "label": row.get("label"),
        "group_id": row.get("group_id"),
        "video_path": str(video_path),
        "frames_dir": str(video_out_dir),
        "status": status,
        "source_fps": float(source_fps),
        "target_fps": float(args.target_fps),
        "source_frames": int(source_frames),
        "output_frames": int(out_frames),
        "detected_frames": int(detected_frames),
        "reused_frames": int(reused_frames),
        "fail_frames": int(fail_frames),
        "fail_rate": float(fail_rate),
        "face_valid_rate": float(1.0 - fail_rate),
        "multi_face_frames": int(multi_face_frames),
        "reinit_count": int(max(len(tracks) - 1, 0)),
        "track_selection": "dominant_after_start",
        "face_valid_path": str(face_valid_path),
        "output_size": int(args.output_size),
        **track_stats,
    }


def process_video(row: pd.Series, face_mesh: Any, out_dir: Path, face_valid_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.track_selection == "dominant_after_start":
        return process_video_dominant_track(row, face_mesh, out_dir, face_valid_dir, args)
    return process_video_online_iou(row, face_mesh, out_dir, face_valid_dir, args)


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("preprocess_faces_mediapipe", args.log_dir)
    logger.info("Log file: %s", log_path)
    log_args(logger, args)
    df = pd.read_csv(args.metadata)
    if "video_id" not in df.columns:
        df["video_id"] = df["video_path"].astype(str).map(lambda x: Path(x).stem)
    duplicate_rows = int(df.duplicated("video_id").sum())
    if duplicate_rows:
        df = df.drop_duplicates("video_id", keep="first").copy()
        logger.warning("Dropped %d duplicate video_id rows before face preprocessing.", duplicate_rows)
    if args.limit:
        df = df.head(args.limit)
    out_dir = ensure_dir(Path(args.out_root) / args.dataset)
    manifest_dir = ensure_dir(Path(args.manifest_root) / args.dataset)
    face_valid_dir = ensure_dir(manifest_dir / "face_valid")
    rows: list[dict[str, Any]] = []
    logger.info("Running face preprocessing for %d clips", len(df))
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=args.max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as face_mesh:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Face preprocessing/{args.dataset}"):
            result = process_video(row, face_mesh, out_dir, face_valid_dir, args)
            rows.append(result)
            if result["status"] != "ok":
                logger.warning(
                    "video_id=%s status=%s output_frames=%s fail_rate=%.4f",
                    result["video_id"],
                    result["status"],
                    result.get("output_frames", 0),
                    float(result.get("fail_rate", 0.0)),
                )
            else:
                logger.info(
                    "video_id=%s status=%s output_frames=%s fail_rate=%.4f",
                    result["video_id"],
                    result["status"],
                    result.get("output_frames", 0),
                    float(result.get("fail_rate", 0.0)),
                )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(manifest_dir / "preprocess_manifest.csv", index=False)
    logger.info("Manifest written to %s", manifest_dir / "preprocess_manifest.csv")
    logger.info("Status counts: %s", manifest["status"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
