from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import ensure_dir, log_args, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract optical flow from preprocessed face frames into .npz float16 files.")
    parser.add_argument("--manifest", required=True, help="preprocess_manifest.csv from preprocess_faces_mediapipe.py")
    parser.add_argument("--dataset", default="dolos", choices=["dolos"])
    parser.add_argument("--out-root", default="data/processed/optflow")
    parser.add_argument("--clip-value", type=float, default=20.0)
    parser.add_argument("--method", default="tvl1", choices=["tvl1", "farneback"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--log-dir", default="outputs/logs")
    return parser.parse_args()


def make_flow_estimator(method: str) -> Any:
    if method == "tvl1":
        try:
            return cv2.optflow.DualTVL1OpticalFlow_create()
        except AttributeError:
            try:
                return cv2.optflow.createOptFlow_DualTVL1()
            except AttributeError:
                return None
    return None


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read frame: {path}")
    return image


def compute_flow(prev: np.ndarray, nxt: np.ndarray, estimator: Any) -> np.ndarray:
    if estimator is not None:
        flow = estimator.calc(prev, nxt, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(
            prev,
            nxt,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
    return flow.astype(np.float32)


def extract_video_flow(frames_dir: Path, estimator: Any, clip_value: float) -> tuple[np.ndarray, np.ndarray]:
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in {frames_dir}")
    timestamps = np.arange(len(frame_paths), dtype=np.float32) / 25.0
    if len(frame_paths) == 1:
        frame = read_gray(frame_paths[0])
        flow = np.zeros((1, 2, frame.shape[0], frame.shape[1]), dtype=np.float16)
        return flow, timestamps

    flows: list[np.ndarray] = []
    prev = read_gray(frame_paths[0])
    for frame_path in frame_paths[1:]:
        nxt = read_gray(frame_path)
        flow_hw2 = compute_flow(prev, nxt, estimator)
        flow_hw2 = np.clip(flow_hw2, -clip_value, clip_value)
        flows.append(np.moveaxis(flow_hw2, -1, 0).astype(np.float16))
        prev = nxt
    flows.append(flows[-1].copy())
    return np.stack(flows, axis=0), timestamps


def process_row(row: pd.Series, out_dir: Path, estimator: Any, clip_value: float, overwrite: bool) -> dict[str, Any]:
    video_id = str(row["video_id"])
    frames_dir = Path(str(row["frames_dir"]))
    out_path = out_dir / f"{video_id}.npz"
    if out_path.exists() and not overwrite:
        return {"video_id": video_id, "frames_dir": str(frames_dir), "flow_path": str(out_path), "status": "exists"}
    ensure_dir(out_path.parent)
    try:
        flow, timestamps = extract_video_flow(frames_dir, estimator, clip_value)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        with tmp_path.open("wb") as f:
            np.savez_compressed(f, flow=flow, timestamps=timestamps)
        tmp_path.replace(out_path)
    except Exception as exc:  # noqa: BLE001 - manifest should capture per-video failures.
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink()
        return {
            "video_id": video_id,
            "frames_dir": str(frames_dir),
            "flow_path": str(out_path),
            "status": "failed",
            "error": str(exc),
        }
    return {
        "video_id": video_id,
        "frames_dir": str(frames_dir),
        "flow_path": str(out_path),
        "status": "ok",
        "flow_frames": int(flow.shape[0]),
        "height": int(flow.shape[2]),
        "width": int(flow.shape[3]),
        "dtype": str(flow.dtype),
    }


def process_record(record: dict[str, Any]) -> dict[str, Any]:
    # Each process owns its OpenCV estimator; TV-L1 objects are not picklable.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    estimator = make_flow_estimator(str(record["method"]))
    row = pd.Series(record["row"])
    return process_row(
        row,
        Path(str(record["out_dir"])),
        estimator,
        float(record["clip_value"]),
        bool(record["overwrite"]),
    )


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("extract_optical_flow", args.log_dir)
    logger.info("Log file: %s", log_path)
    log_args(logger, args)
    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["status"].isin(["ok", "exists"])].copy()
    duplicate_rows = int(manifest.duplicated("video_id").sum()) if "video_id" in manifest.columns else 0
    if duplicate_rows:
        manifest = manifest.drop_duplicates("video_id", keep="first").copy()
        logger.warning("Dropped %d duplicate video_id rows before optical-flow extraction.", duplicate_rows)
    if args.limit:
        manifest = manifest.head(args.limit)
    out_dir = ensure_dir(Path(args.out_root) / args.dataset)
    estimator = make_flow_estimator(args.method)
    tvl1_available = estimator is not None
    if not tvl1_available and args.method == "tvl1":
        logger.warning("TV-L1 unavailable in OpenCV build; falling back to Farneback.")
    logger.info("Extracting optical flow for %d clips with workers=%d", len(manifest), args.workers)
    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc=f"OptFlow/{args.dataset}"):
            result = process_row(row, out_dir, estimator, args.clip_value, args.overwrite)
            rows.append(result)
            if result["status"] == "failed":
                logger.error("video_id=%s status=%s flow_path=%s error=%s", result["video_id"], result["status"], result["flow_path"], result.get("error", ""))
            else:
                logger.info("video_id=%s status=%s flow_path=%s", result["video_id"], result["status"], result["flow_path"])
    else:
        records = [
            {
                "row": row.to_dict(),
                "out_dir": str(out_dir),
                "method": args.method,
                "clip_value": args.clip_value,
                "overwrite": args.overwrite,
            }
            for _, row in manifest.iterrows()
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for result in tqdm(executor.map(process_record, records), total=len(records), desc=f"OptFlow/{args.dataset}"):
                rows.append(result)
                if result["status"] == "failed":
                    logger.error("video_id=%s status=%s flow_path=%s error=%s", result["video_id"], result["status"], result["flow_path"], result.get("error", ""))
                else:
                    logger.info("video_id=%s status=%s flow_path=%s", result["video_id"], result["status"], result["flow_path"])
    out_manifest = pd.DataFrame(rows)
    out_manifest.to_csv(out_dir / "optflow_manifest.csv", index=False)
    logger.info("Manifest written to %s", out_dir / "optflow_manifest.csv")
    logger.info("Status counts: %s", out_manifest["status"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
