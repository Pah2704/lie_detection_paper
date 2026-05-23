from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils import ensure_dir, log_args, setup_logger


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 16 kHz mono WAV audio from videos.")
    parser.add_argument("--metadata", default=None, help="CSV containing video_id and video_path.")
    parser.add_argument("--video-dir", default=None, help="Fallback directory to scan when metadata is omitted.")
    parser.add_argument("--dataset", default="dolos", choices=["dolos"])
    parser.add_argument("--out-root", default="data/processed/audio")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--ffmpeg", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-dir", default="outputs/logs")
    return parser.parse_args()


def load_rows(metadata: str | None, video_dir: str | None) -> pd.DataFrame:
    if metadata:
        df = pd.read_csv(metadata)
        if "video_id" not in df.columns:
            df["video_id"] = df["video_path"].astype(str).map(lambda x: Path(x).stem)
        return df
    if not video_dir:
        raise ValueError("Either --metadata or --video-dir is required.")
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(video_dir).rglob("*")):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            rows.append({"video_id": path.stem, "video_path": str(path)})
    return pd.DataFrame(rows)


def deduplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "video_id" not in df.columns:
        return df, 0
    duplicate_rows = int(df.duplicated("video_id").sum())
    if duplicate_rows:
        df = df.drop_duplicates("video_id", keep="first").copy()
    return df, duplicate_rows


def ffmpeg_binary(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise FileNotFoundError("ffmpeg not found on PATH; pass --ffmpeg.")


def extract_one(ffmpeg: str, video_path: Path, out_path: Path, sample_rate: int, overwrite: bool) -> dict[str, Any]:
    if out_path.exists() and not overwrite:
        return {"status": "exists", "audio_path": str(out_path)}
    ensure_dir(out_path.parent)
    cmd = [
        ffmpeg,
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-loglevel",
        "error",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        return {"status": "failed", "audio_path": str(out_path), "error": proc.stderr.strip()}
    return {"status": "ok", "audio_path": str(out_path)}


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("extract_audio", args.log_dir)
    logger.info("Log file: %s", log_path)
    log_args(logger, args)
    df = load_rows(args.metadata, args.video_dir)
    df, duplicate_rows = deduplicate_rows(df)
    if duplicate_rows:
        logger.warning("Dropped %d duplicate video_id rows before audio extraction.", duplicate_rows)
    if args.limit:
        df = df.head(args.limit)
    out_dir = ensure_dir(Path(args.out_root) / args.dataset)
    ffmpeg = ffmpeg_binary(args.ffmpeg)
    logger.info("Extracting audio for %d clips with ffmpeg=%s", len(df), ffmpeg)

    rows: list[dict[str, Any]] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting audio/{args.dataset}"):
        video_path = Path(str(row["video_path"]))
        video_id = str(row["video_id"])
        out_path = out_dir / f"{video_id}.wav"
        result = extract_one(ffmpeg, video_path, out_path, args.sample_rate, args.overwrite)
        record = {"video_id": video_id, "video_path": str(video_path), **result}
        rows.append(record)
        if result["status"] == "failed":
            logger.error("video_id=%s status=%s audio_path=%s error=%s", video_id, result["status"], result["audio_path"], result.get("error", ""))
        else:
            logger.info("video_id=%s status=%s audio_path=%s", video_id, result["status"], result["audio_path"])
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "audio_manifest.csv", index=False)
    logger.info("Manifest written to %s", out_dir / "audio_manifest.csv")
    logger.info("Status counts: %s", manifest["status"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
