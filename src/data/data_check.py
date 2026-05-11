from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.utils import ensure_dir, write_json


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
LABEL_CANDIDATES = (
    "label",
    "class",
    "truth",
    "lie",
    "is_lie",
    "is_deceptive",
    "deception",
    "truthfulness",
    "veracity",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Bag-of-Lies files and build metadata.")
    parser.add_argument("--data-root", default="data/raw/BagOfLies", help="Extracted BagOfLies root.")
    parser.add_argument("--zip", default="data/BagOfLies.zip", help="Archive path for pre-extraction checks.")
    parser.add_argument("--annotation", default=None, help="Annotation CSV path.")
    parser.add_argument("--label-column", default=None, help="Annotation label column, if auto-detection fails.")
    parser.add_argument("--out-dir", default="outputs/metrics/data_check", help="Output report directory.")
    parser.add_argument("--skip-video-probe", action="store_true", help="Skip OpenCV video probing.")
    return parser.parse_args()


def archive_summary(zip_path: Path) -> dict[str, Any]:
    if not zip_path.exists():
        return {"exists": False}
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        videos = [n for n in names if Path(n).suffix.lower() in VIDEO_EXTS]
        users = sorted({m.group(1) for n in names if (m := re.search(r"Finalised/(User_\d+)/", n))})
        encrypted_count = sum(1 for info in zf.infolist() if info.flag_bits & 0x1)
        return {
            "exists": True,
            "path": str(zip_path),
            "entries": len(names),
            "video_entries": len(videos),
            "users": users,
            "num_users": len(users),
            "encrypted_entries": encrypted_count,
            "has_annotations_csv": any(n.endswith("Annotations.csv") for n in names),
        }


def parse_subject_run(path: Path) -> tuple[str | None, str | None]:
    parts = path.as_posix().split("/")
    subject = next((p for p in parts if re.fullmatch(r"User_\d+", p)), None)
    run = next((p for p in parts if re.fullmatch(r"run_\d+", p)), None)
    return subject, run


def label_from_path(path: Path) -> int | None:
    parts = {p.lower() for p in path.parts}
    name = path.name.lower()
    if "deceptive" in parts or "lie" in name:
        return 1
    if "truthful" in parts or "truth" in name:
        return 0
    return None


def transcript_for_video(path: Path) -> str | None:
    try:
        rel = path.relative_to(path.parents[2])
    except ValueError:
        return None
    root = path.parents[2]
    candidate = root / "Transcription" / rel.relative_to("Clips")
    candidate = candidate.with_suffix(".txt")
    return str(candidate) if candidate.exists() else None


def parse_readme_tables(data_root: Path) -> pd.DataFrame:
    readme = data_root / "README.txt"
    if not readme.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for line in readme.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line.startswith("| trial_"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        video_id, role_trial, ruling, link = cells[:4]
        role = None
        trial_name = role_trial
        if "/" in role_trial:
            role, trial_name = [part.strip() for part in role_trial.split("/", 1)]
        rows.append(
            {
                "video_id": video_id,
                "role_trial_name": role_trial,
                "role": role,
                "trial_name": re.sub(r"\s+", " ", trial_name).strip(),
                "ruling": ruling,
                "source_link": link,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["video_id"], keep="first")
    df["group_id"] = df["trial_name"].fillna(df["video_id"])
    return df


def find_videos(data_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not data_root.exists():
        return pd.DataFrame(rows)
    for path in sorted(data_root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTS:
            continue
        subject, run = parse_subject_run(path)
        rows.append(
            {
                "video_id": path.name if path.name.startswith("trial_") else f"{subject}_{run}" if subject and run else path.stem,
                "subject_id": subject,
                "run_id": run,
                "video_path": str(path),
                "label": label_from_path(path),
                "transcript_path": transcript_for_video(path),
                "file_size_bytes": path.stat().st_size,
            }
        )
    videos = pd.DataFrame(rows)
    readme_rows = parse_readme_tables(data_root)
    if not videos.empty and not readme_rows.empty:
        videos = videos.merge(readme_rows, on="video_id", how="left")
    return videos


def probe_videos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    try:
        import cv2
    except ImportError:
        return probe_videos_ffprobe(df)

    probed: list[dict[str, Any]] = []
    for row in tqdm(df.to_dict("records"), desc="Probing videos"):
        path = row["video_path"]
        cap = cv2.VideoCapture(path)
        readable = bool(cap.isOpened())
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if readable else 0.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if readable else 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if readable else 0
        cap.release()
        row.update(
            {
                "video_readable": readable,
                "fps": fps,
                "num_frames": frames,
                "duration_sec": frames / fps if fps > 0 else 0.0,
                "width": width,
                "height": height,
                "video_error": "" if readable and frames > 0 else "unreadable_or_empty",
            }
        )
        probed.append(row)
    return pd.DataFrame(probed)


def _parse_fps(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def probe_videos_ffprobe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    probed: list[dict[str, Any]] = []
    for row in tqdm(df.to_dict("records"), desc="Probing videos with ffprobe"):
        path = row["video_path"]
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            path,
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            payload = json.loads(result.stdout)
            stream = payload.get("streams", [{}])[0]
            fps = _parse_fps(stream.get("avg_frame_rate"))
            duration = float(stream.get("duration") or 0.0)
            try:
                frames = int(stream.get("nb_frames") or 0)
            except (TypeError, ValueError):
                frames = 0
            if frames <= 0 and fps > 0:
                frames = int(round(duration * fps))
            row.update(
                {
                    "video_readable": True,
                    "fps": fps,
                    "num_frames": frames,
                    "duration_sec": duration,
                    "width": int(stream.get("width") or 0),
                    "height": int(stream.get("height") or 0),
                    "video_error": "",
                }
            )
        except Exception as exc:
            row.update(
                {
                    "video_readable": False,
                    "fps": 0.0,
                    "num_frames": 0,
                    "duration_sec": 0.0,
                    "width": 0,
                    "height": 0,
                    "video_error": f"ffprobe_error:{type(exc).__name__}",
                }
            )
        probed.append(row)
    return pd.DataFrame(probed)


def read_annotation(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def detect_label_column(annotation: pd.DataFrame, explicit: str | None) -> str | None:
    if annotation.empty:
        return None
    if explicit:
        if explicit not in annotation.columns:
            raise ValueError(f"Label column {explicit!r} not found. Columns: {list(annotation.columns)}")
        return explicit
    lower_map = {c.lower().strip(): c for c in annotation.columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    for col in annotation.columns:
        values = set(str(v).strip().lower() for v in annotation[col].dropna().unique())
        if values and values <= {"truth", "true", "lie", "false", "0", "1", "yes", "no"}:
            return col
    return None


def normalize_label(value: Any) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"1", "lie", "lies", "false", "deceptive", "deception", "yes", "l"}:
        return 1
    if text in {"0", "truth", "true", "truthful", "nonlie", "non-lie", "no", "t"}:
        return 0
    return None


def annotation_with_keys(annotation: pd.DataFrame, label_col: str | None) -> pd.DataFrame:
    if annotation.empty:
        return annotation
    ann = annotation.copy()
    lower_cols = {c.lower().strip(): c for c in ann.columns}
    subject_col = next((lower_cols[c] for c in ("subject_id", "subject", "user", "user_id", "participant") if c in lower_cols), None)
    run_col = next((lower_cols[c] for c in ("run_id", "run", "trial", "trial_id", "video_id") if c in lower_cols), None)

    if subject_col:
        ann["subject_id"] = ann[subject_col].astype(str).str.extract(r"(\d+)", expand=False).map(lambda x: f"User_{x}" if pd.notna(x) else None)
    if run_col:
        ann["run_id"] = ann[run_col].astype(str).str.extract(r"(\d+)", expand=False).map(lambda x: f"run_{x}" if pd.notna(x) else None)
    if "subject_id" in ann.columns and "run_id" in ann.columns:
        ann["video_id"] = ann["subject_id"].astype(str) + "_" + ann["run_id"].astype(str)
    elif "id" in lower_cols:
        ann["video_id"] = ann[lower_cols["id"]].astype(str)
    if label_col:
        ann["label"] = ann[label_col].map(normalize_label)
    return ann


def merge_annotation(videos: pd.DataFrame, annotation: pd.DataFrame) -> pd.DataFrame:
    if videos.empty or annotation.empty or "video_id" not in annotation.columns:
        return apply_known_dataset_fixes(videos)
    keep_cols = [c for c in ("video_id", "label") if c in annotation.columns and c not in videos.columns]
    keep_cols = ["video_id", *keep_cols]
    extra_cols = [c for c in annotation.columns if c not in videos.columns and c not in keep_cols]
    merged = videos.merge(annotation[keep_cols + extra_cols], on="video_id", how="left")
    if "label" not in merged and "label_x" in merged:
        merged["label"] = merged["label_x"]
    return apply_known_dataset_fixes(merged)


def apply_known_dataset_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """Patch documented quirks in supported public datasets."""
    if df.empty or "video_id" not in df.columns:
        return df
    fixed = df.copy()

    # Real-life Trial README has a table typo: trial_truth_008 is listed twice
    # and trial_truth_009 is omitted. The surrounding sequence and source link
    # indicate trial_truth_009 belongs to the Jodi Arias truthful witness group.
    mask = fixed["video_id"].astype(str).eq("trial_truth_009.mp4")
    if mask.any() and ("group_id" not in fixed.columns or fixed.loc[mask, "group_id"].isna().any()):
        defaults = {
            "role_trial_name": "Witness / Jodi Arias",
            "role": "Witness",
            "trial_name": "Jodi Arias",
            "ruling": "Guilty",
            "source_link": "https://www.youtube.com/watch?v=FT28C3b5GKA",
            "group_id": "Jodi Arias",
        }
        for column, value in defaults.items():
            if column not in fixed.columns:
                fixed[column] = pd.NA
            fixed.loc[mask & fixed[column].isna(), column] = value
    return fixed


def write_reports(df: pd.DataFrame, annotation: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    ensure_dir(out_dir)
    if not df.empty:
        df.to_csv(out_dir / "metadata.csv", index=False)
    if not annotation.empty:
        annotation.head(50).to_csv(out_dir / "annotation_preview.csv", index=False)

    report = dict(summary)
    report["extracted_videos"] = int(len(df))
    report["extracted_subjects"] = sorted(df["subject_id"].dropna().unique().tolist()) if "subject_id" in df else []
    report["num_extracted_subjects"] = len(report["extracted_subjects"])
    if "group_id" in df:
        report["groups"] = sorted(df["group_id"].dropna().unique().tolist())
        report["num_groups"] = int(df["group_id"].dropna().nunique())
    if "label" in df:
        report["label_counts"] = {str(k): int(v) for k, v in Counter(df["label"].dropna().astype(int)).items()}
    if "video_readable" in df:
        readable = df["video_readable"].fillna(False).astype(bool)
        report["readable_videos"] = int(readable.sum())
        report["unreadable_videos"] = int((~readable).sum())
    if not annotation.empty:
        report["annotation_columns"] = list(annotation.columns)
        report["annotation_rows"] = int(len(annotation))
    write_json(out_dir / "data_check_summary.json", report)

    if not df.empty and "subject_id" in df:
        by_subject = defaultdict(Counter)
        for row in df.to_dict("records"):
            subject = row.get("subject_id")
            label = row.get("label")
            if subject:
                key = "missing_label" if pd.isna(label) else str(int(label))
                by_subject[subject][key] += 1
        subject_rows = [{"subject_id": s, **dict(counts)} for s, counts in sorted(by_subject.items())]
        pd.DataFrame(subject_rows).fillna(0).to_csv(out_dir / "subject_label_counts.csv", index=False)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    annotation_path = Path(args.annotation) if args.annotation else data_root / "Annotations.csv"
    out_dir = Path(args.out_dir)

    summary = {"archive": archive_summary(Path(args.zip)), "data_root": str(data_root)}
    videos = find_videos(data_root)
    if not args.skip_video_probe:
        videos = probe_videos(videos)

    annotation = read_annotation(annotation_path)
    label_col = detect_label_column(annotation, args.label_column)
    annotation = annotation_with_keys(annotation, label_col)
    metadata = merge_annotation(videos, annotation)

    summary["annotation_path"] = str(annotation_path)
    summary["label_column"] = label_col
    write_reports(metadata, annotation, summary, out_dir)

    print(f"Wrote data check outputs to {out_dir}")
    print(f"Videos found: {len(metadata)}")
    if not metadata.empty and "label" in metadata:
        print("Label counts:")
        print(metadata["label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
