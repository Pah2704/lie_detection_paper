from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils import ensure_dir, log_args, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download DOLOS clips from dolos_timestamps.csv using yt-dlp/youtube-dl.")
    parser.add_argument("--timestamps", default="data/DOLOS/dolos_timestamps.csv")
    parser.add_argument("--out-dir", default="data/raw/dolos/videos")
    parser.add_argument("--cookies", default=None, help="Optional browser cookies file for yt-dlp.")
    parser.add_argument("--downloader", default=None, help="yt-dlp or youtube-dl binary path.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-bytes", type=int, default=1024, help="Treat smaller existing files as corrupt and re-download them.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-requests", type=float, default=0.0)
    parser.add_argument("--log-dir", default="outputs/logs")
    return parser.parse_args()


def find_downloader(explicit: str | None) -> str:
    if explicit:
        return explicit
    for name in ("yt-dlp", "youtube-dl"):
        found = shutil.which(name)
        if found:
            return found
    raise FileNotFoundError("Install yt-dlp or pass --downloader. Example: pip install yt-dlp")


def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def parse_timestamp(value: Any) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError("Empty timestamp")
    parts = [part.strip() for part in text.split(":") if part.strip()]
    if len(parts) == 1:
        return int(float(parts[0]) * 60)
    if len(parts) == 2:
        minutes, seconds = (int(part) for part in parts)
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = (int(part) for part in parts)
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Unsupported timestamp: {value}")


def format_timestamp(total_seconds: int) -> str:
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def build_cmd(downloader: str, row: pd.Series, out_path: Path, cookies: str | None, overwrite: bool, sleep_requests: float) -> list[str]:
    start_seconds = parse_timestamp(row["start_time"])
    end_seconds = parse_timestamp(row["end_time"])
    if end_seconds <= start_seconds:
        raise ValueError(f"Invalid section for {row['file_name']}: {row['start_time']} -> {row['end_time']}")
    start = format_timestamp(start_seconds)
    end = format_timestamp(end_seconds)
    url = youtube_url(str(row["YT_Video_ID"]))
    cmd = [
        downloader,
        "--quiet",
        "--no-warnings",
        "--download-sections",
        f"*{start}-{end}",
        "-f",
        "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_path),
    ]
    if not overwrite:
        cmd.append("--no-overwrites")
    if cookies:
        cmd.extend(["--cookies", cookies])
    if sleep_requests > 0:
        cmd.extend(["--sleep-requests", str(sleep_requests)])
    cmd.append(url)
    return cmd


def download_one(
    downloader: str,
    row: pd.Series,
    out_dir: Path,
    cookies: str | None,
    overwrite: bool,
    sleep_requests: float,
    min_bytes: int,
) -> dict[str, Any]:
    video_id = str(row["file_name"])
    out_path = out_dir / f"{video_id}.mp4"
    if out_path.exists() and not overwrite:
        if out_path.stat().st_size >= min_bytes:
            return {"video_id": video_id, "video_path": str(out_path), "status": "exists"}
        out_path.unlink()
    try:
        cmd = build_cmd(downloader, row, out_path, cookies, overwrite, sleep_requests)
    except ValueError as exc:
        return {
            "video_id": video_id,
            "video_path": str(out_path),
            "status": "failed",
            "error": str(exc),
        }
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size < min_bytes:
        return {
            "video_id": video_id,
            "video_path": str(out_path),
            "status": "failed",
            "error": (proc.stderr or proc.stdout).strip()[-1000:],
        }
    return {"video_id": video_id, "video_path": str(out_path), "status": "ok"}


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("download_dolos_videos", args.log_dir)
    logger.info("Log file: %s", log_path)
    log_args(logger, args)
    timestamps = pd.read_csv(args.timestamps)
    if args.limit:
        timestamps = timestamps.head(args.limit)
    out_dir = ensure_dir(args.out_dir)
    downloader = find_downloader(args.downloader)
    logger.info("Downloading %d DOLOS clips with %s", len(timestamps), downloader)
    rows: list[dict[str, Any]] = []
    for _, row in tqdm(timestamps.iterrows(), total=len(timestamps), desc="Downloading DOLOS clips"):
        result = download_one(
            downloader,
            row,
            out_dir,
            args.cookies,
            args.overwrite,
            args.sleep_requests,
            args.min_bytes,
        )
        rows.append(result)
        if result["status"] == "failed":
            logger.error("video_id=%s status=%s path=%s error=%s", result["video_id"], result["status"], result["video_path"], result.get("error", ""))
        else:
            logger.info("video_id=%s status=%s path=%s", result["video_id"], result["status"], result["video_path"])
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "download_manifest.csv", index=False)
    logger.info("Manifest written to %s", out_dir / "download_manifest.csv")
    logger.info("Status counts: %s", manifest["status"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
