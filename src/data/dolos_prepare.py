from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from src.utils import ensure_dir, log_args, setup_logger, write_json


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IDENTITY_COLUMNS = ("subject_id", "person_id", "speaker_id", "person_name")
LABEL_COLUMNS = ("label", "truth", "veracity", "class", "deception", "is_lie")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare DOLOS metadata and 3-fold group-aware splits.")
    parser.add_argument("--video-dir", default="data/raw/dolos/videos")
    parser.add_argument("--metadata", default="data/DOLOS/dolos_timestamps.csv", help="Optional CSV/XLSX metadata file.")
    parser.add_argument("--protocol-dir", default="data/DOLOS/Training_Protocols")
    parser.add_argument("--out-dir", default="data/processed/splits/dolos")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-bytes", type=int, default=1024)
    parser.add_argument("--duration-audit", action="store_true", help="Filter videos whose raw duration does not match timestamp duration.")
    parser.add_argument("--max-duration-slack", type=float, default=5.0, help="Allowed seconds above expected timestamp duration.")
    parser.add_argument("--ffprobe", default=None)
    parser.add_argument("--log-dir", default="outputs/logs")
    return parser.parse_args()


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def canonical_video_id(value: Any) -> str:
    return Path(str(value).strip()).stem.strip()


def normalize_label(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, (float, np.floating)) and int(value) in (0, 1):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "lie", "liar", "deceptive", "deception", "false", "fake"}:
        return 1
    if text in {"0", "truth", "true", "truthful", "nonlie", "non_lie", "honest"}:
        return 0
    if "lie" in text and "non" not in text:
        return 1
    if "truth" in text or re.search(r"\btrue\b", text):
        return 0
    return None


def parse_timestamp_seconds(value: Any) -> int:
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


def parse_video_name(path: Path) -> dict[str, Any]:
    stem = path.stem
    normalized = stem.replace(" ", "_")
    ep_match = re.search(r"EP[_-]?(\d+)", normalized, flags=re.IGNORECASE)
    host = normalized.split("_", 1)[0] if "_" in normalized else ""
    label = normalize_label(normalized)
    episode = ep_match.group(1) if ep_match else ""
    group_id = f"{host}_EP{episode}" if host and episode else host or stem
    return {
        "video_id": stem,
        "video_path": str(path),
        "filename_label": label,
        "host": host,
        "episode": episode,
        "episode_group_id": group_id,
    }


def scan_videos(video_dir: Path, min_bytes: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(video_dir.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        info = parse_video_name(path)
        size_bytes = path.stat().st_size
        info["size_bytes"] = int(size_bytes)
        info["is_corrupt"] = bool(size_bytes < min_bytes)
        rows.append(info)
    if not rows:
        raise FileNotFoundError(f"No videos found under {video_dir}")
    return pd.DataFrame(rows)


def load_metadata(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df = df.rename(columns={col: normalize_col(col) for col in df.columns})
    return df


def metadata_with_video_id(metadata_df: pd.DataFrame) -> pd.DataFrame:
    meta = metadata_df.copy()
    if "file_name" in meta.columns and "video_id" not in meta.columns:
        meta["video_id"] = meta["file_name"].map(canonical_video_id)
    if "video_id" not in meta.columns:
        path_cols = [c for c in meta.columns if "path" in c or "file" in c or "video" in c]
        if path_cols:
            meta["video_id"] = meta[path_cols[0]].map(canonical_video_id)
    if "video_id" not in meta.columns:
        raise ValueError("Metadata must contain a video_id/path/file/video column.")
    meta["video_id"] = meta["video_id"].map(canonical_video_id)
    return meta


def merge_metadata(video_df: pd.DataFrame, metadata_df: pd.DataFrame | None) -> pd.DataFrame:
    df = video_df.copy()
    df["video_id"] = df["video_id"].map(canonical_video_id)
    if metadata_df is None or metadata_df.empty:
        df["label"] = df["filename_label"]
        return df

    meta = metadata_with_video_id(metadata_df)

    merged = df.merge(meta, on="video_id", how="left", suffixes=("", "_meta"))
    label_col = next((c for c in LABEL_COLUMNS if c in merged.columns), None)
    if label_col is not None:
        merged["label"] = merged[label_col].map(normalize_label)
        merged["label"] = merged["label"].fillna(merged["filename_label"])
    else:
        merged["label"] = merged["filename_label"]
    return merged


def write_metadata_quality_reports(
    out_dir: Path,
    videos: pd.DataFrame,
    metadata: pd.DataFrame | None,
    df: pd.DataFrame,
    logger,
) -> tuple[int, int]:
    if metadata is None or metadata.empty:
        missing_metadata_rows = 0
    else:
        meta = metadata_with_video_id(metadata)
        video_ids = set(videos["video_id"].map(canonical_video_id))
        missing = meta[~meta["video_id"].isin(video_ids)].copy()
        missing_metadata_rows = int(len(missing))
        missing_path = out_dir / "missing_timestamps.csv"
        if missing_metadata_rows:
            missing.to_csv(missing_path, index=False)
            logger.warning("Metadata rows without downloaded video: %d (%s)", missing_metadata_rows, missing_path)
        elif missing_path.exists():
            missing_path.unlink()

    duplicates = df[df.duplicated("video_id", keep=False)].sort_values("video_id").copy()
    duplicate_rows = int(len(duplicates))
    duplicate_path = out_dir / "duplicate_video_ids.csv"
    if duplicate_rows:
        duplicates.to_csv(duplicate_path, index=False)
        logger.warning("Duplicate video_id rows before deduplication: %d (%s)", duplicate_rows, duplicate_path)
    elif duplicate_path.exists():
        duplicate_path.unlink()
    return missing_metadata_rows, duplicate_rows


def ffprobe_binary(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("ffprobe")
    if found:
        return found
    raise FileNotFoundError("ffprobe not found on PATH; pass --ffprobe or disable --duration-audit.")


def probe_duration_seconds(ffprobe: str, path: Path) -> float:
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return float("nan")
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return float("nan")


def audit_and_filter_durations(
    df: pd.DataFrame,
    out_dir: Path,
    ffprobe: str,
    max_duration_slack: float,
    logger,
) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, Any]] = []
    keep_mask: list[bool] = []
    for _, row in df.iterrows():
        video_path = Path(str(row["video_path"]))
        duration = probe_duration_seconds(ffprobe, video_path)
        try:
            start_seconds = parse_timestamp_seconds(row["start_time"])
            end_seconds = parse_timestamp_seconds(row["end_time"])
            expected_duration = end_seconds - start_seconds
        except Exception:
            start_seconds = None
            end_seconds = None
            expected_duration = float("nan")
        too_long = bool(
            np.isfinite(duration)
            and np.isfinite(expected_duration)
            and expected_duration > 0
            and duration > expected_duration + max_duration_slack
        )
        too_short = bool(
            np.isfinite(duration)
            and np.isfinite(expected_duration)
            and expected_duration > 0
            and duration + 1.0 < expected_duration
        )
        rows.append(
            {
                "video_id": row["video_id"],
                "video_path": str(video_path),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "raw_duration_seconds": duration,
                "expected_duration_seconds": expected_duration,
                "too_long": too_long,
                "too_short": too_short,
            }
        )
        keep_mask.append(not too_long and not too_short)

    audit = pd.DataFrame(rows)
    audit_path = out_dir / "raw_video_duration_audit.csv"
    audit.to_csv(audit_path, index=False)
    excluded = audit[~pd.Series(keep_mask, index=audit.index)].copy()
    excluded_path = out_dir / "excluded_duration_mismatch.csv"
    if not excluded.empty:
        excluded.to_csv(excluded_path, index=False)
        logger.warning("Duration audit excluded %d videos (%s)", len(excluded), excluded_path)
    elif excluded_path.exists():
        excluded_path.unlink()
    logger.info("Raw duration audit written to %s", audit_path)
    return df.loc[keep_mask].copy(), int((~pd.Series(keep_mask)).sum())


def load_protocol(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 2:
        raise ValueError(f"Protocol must have at least 2 columns: {path}")
    df = raw.iloc[:, :3].copy()
    columns = ["video_id", "label", "gender"][: df.shape[1]]
    df.columns = columns
    df["video_id"] = df["video_id"].map(canonical_video_id)
    df["label"] = df["label"].map(normalize_label)
    return df.dropna(subset=["video_id", "label"]).drop_duplicates("video_id", keep="first").copy()


def official_protocols_available(protocol_dir: Path, folds: int) -> bool:
    return all((protocol_dir / f"train_fold{i}.csv").exists() and (protocol_dir / f"test_fold{i}.csv").exists() for i in range(1, folds + 1))


def make_official_folds(
    df: pd.DataFrame,
    protocol_dir: Path,
    out_dir: Path,
    folds: int,
    val_fraction: float,
    seed: int,
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, Any]]:
    by_id = df.drop_duplicates("video_id").set_index("video_id")
    output: dict[str, dict[str, pd.DataFrame]] = {}
    missing_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    unassigned_rows: list[dict[str, Any]] = []
    available_ids = set(by_id.index)
    for fold_idx in range(1, folds + 1):
        train_proto = load_protocol(protocol_dir / f"train_fold{fold_idx}.csv")
        test_proto = load_protocol(protocol_dir / f"test_fold{fold_idx}.csv")
        overlap = sorted(set(train_proto["video_id"]) & set(test_proto["video_id"]))
        if overlap:
            overlap_rows.extend({"fold": fold_idx, "video_id": vid, "kept_split": "test"} for vid in overlap)
            train_proto = train_proto[~train_proto["video_id"].isin(overlap)].copy()
        for proto_name, proto_df in (("train", train_proto), ("test", test_proto)):
            missing = sorted(set(proto_df["video_id"]) - set(by_id.index))
            missing_rows.extend({"fold": fold_idx, "split": proto_name, "video_id": vid} for vid in missing)
        assigned_ids = set(train_proto["video_id"]) | set(test_proto["video_id"])
        unassigned = sorted(available_ids - assigned_ids)
        unassigned_rows.extend({"fold": fold_idx, "video_id": vid} for vid in unassigned)
        trainval_df = by_id.reindex(train_proto["video_id"]).dropna(subset=["video_path"]).reset_index()
        test_df = by_id.reindex(test_proto["video_id"]).dropna(subset=["video_path"]).reset_index()
        train_df, val_df = train_val_split(trainval_df, "group_id", val_fraction, seed + fold_idx)
        output[f"fold{fold_idx}"] = {"train": train_df, "val": val_df, "test": test_df}

    report_specs = {
        "official_protocol_missing_downloaded.csv": missing_rows,
        "official_protocol_overlaps_removed.csv": overlap_rows,
        "official_protocol_unassigned_downloaded.csv": unassigned_rows,
    }
    for filename, rows in report_specs.items():
        path = out_dir / filename
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
        elif path.exists():
            path.unlink()

    diagnostics = {
        "official_protocol_missing_downloaded_rows": int(len(missing_rows)),
        "official_protocol_overlaps_removed_rows": int(len(overlap_rows)),
        "official_protocol_unassigned_downloaded_rows": int(len(unassigned_rows)),
        "official_protocol_unassigned_downloaded_unique": int(pd.DataFrame(unassigned_rows)["video_id"].nunique()) if unassigned_rows else 0,
    }
    return output, diagnostics


def choose_group_column(df: pd.DataFrame) -> tuple[str, str]:
    for col in IDENTITY_COLUMNS:
        if col in df.columns and df[col].notna().any():
            return col, f"grouped by identity column: {col}"
    return "episode_group_id", "grouped by episode; subject identity across episodes not verified"


def summarize_split(df: pd.DataFrame, group_col: str) -> dict[str, Any]:
    counts = df["label"].value_counts(dropna=False).to_dict()
    return {
        "num_rows": int(len(df)),
        "num_groups": int(df[group_col].nunique()),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
    }


def train_val_split(
    trainval_df: pd.DataFrame,
    group_col: str,
    val_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_val_splits = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
    groups = trainval_df[group_col].astype(str).to_numpy()
    y = trainval_df["label"].astype(int).to_numpy()
    try:
        splitter = StratifiedGroupKFold(n_splits=n_val_splits, shuffle=True, random_state=seed)
        train_idx, val_idx = next(splitter.split(trainval_df, y, groups))
    except ValueError:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(trainval_df, y, groups))
    return trainval_df.iloc[train_idx].copy(), trainval_df.iloc[val_idx].copy()


def make_folds(df: pd.DataFrame, group_col: str, folds: int, val_fraction: float, seed: int) -> dict[str, dict[str, pd.DataFrame]]:
    y = df["label"].astype(int).to_numpy()
    groups = df[group_col].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    output: dict[str, dict[str, pd.DataFrame]] = {}
    for fold_idx, (trainval_idx, test_idx) in enumerate(splitter.split(df, y, groups), start=1):
        trainval_df = df.iloc[trainval_idx].copy()
        test_df = df.iloc[test_idx].copy()
        train_df, val_df = train_val_split(trainval_df, group_col, val_fraction, seed + fold_idx)
        output[f"fold{fold_idx}"] = {"train": train_df, "val": val_df, "test": test_df}
    return output


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("dolos_prepare", args.log_dir)
    logger.info("Log file: %s", log_path)
    log_args(logger, args)
    out_dir = ensure_dir(args.out_dir)
    videos = scan_videos(Path(args.video_dir), args.min_bytes)
    logger.info("Scanned %d videos from %s", len(videos), args.video_dir)
    metadata = load_metadata(Path(args.metadata) if args.metadata else None)
    logger.info("Metadata source: %s", args.metadata if args.metadata else "none")
    df = merge_metadata(videos, metadata)
    df = df[~df["is_corrupt"]].copy()
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    missing_metadata_rows, duplicate_video_id_rows = write_metadata_quality_reports(out_dir, videos, metadata, df, logger)
    if duplicate_video_id_rows:
        df = df.drop_duplicates("video_id", keep="first").copy()
        logger.info("Deduplicated metadata to one row per video_id: %d rows", len(df))
    duration_excluded_rows = 0
    if args.duration_audit:
        ffprobe = ffprobe_binary(args.ffprobe)
        df, duration_excluded_rows = audit_and_filter_durations(
            df,
            out_dir,
            ffprobe,
            args.max_duration_slack,
            logger,
        )
    group_col, group_note = choose_group_column(df)
    df["group_id"] = df[group_col].astype(str)
    logger.info("Usable rows after filtering: %d", len(df))
    logger.info("Grouping note: %s", group_note)

    if df["label"].nunique() < 2:
        raise ValueError("Need both lie/truth labels after normalization.")

    protocol_dir = Path(args.protocol_dir)
    use_official = official_protocols_available(protocol_dir, args.folds)
    protocol_diagnostics: dict[str, Any] = {}
    if use_official:
        folds, protocol_diagnostics = make_official_folds(df, protocol_dir, out_dir, args.folds, args.val_fraction, args.seed)
    else:
        folds = make_folds(df, "group_id", args.folds, args.val_fraction, args.seed)
    logger.info("Split source: %s", "official protocols" if use_official else "generated stratified group folds")
    summary: dict[str, Any] = {
        "seed": args.seed,
        "folds": args.folds,
        "val_fraction": args.val_fraction,
        "group_column": group_col,
        "group_note": group_note,
        "protocol_source": str(protocol_dir) if use_official else "generated",
        "scanned_videos": int(len(videos)),
        "missing_metadata_rows": missing_metadata_rows,
        "duplicate_video_id_rows": duplicate_video_id_rows,
        "duration_excluded_rows": duration_excluded_rows,
        "usable_rows": int(len(df)),
        "all_label_counts": {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()},
        "splits": {},
    }
    summary.update(protocol_diagnostics)
    df.to_csv(out_dir / "metadata.csv", index=False)
    for fold_name, parts in folds.items():
        summary["splits"][fold_name] = {}
        for split_name, split_df in parts.items():
            split_df = split_df.sort_values("video_id").copy()
            split_df["split"] = split_name
            split_df["fold"] = fold_name
            split_df.to_csv(out_dir / f"{fold_name}_{split_name}.csv", index=False)
            summary["splits"][fold_name][split_name] = summarize_split(split_df, "group_id")
            logger.info("%s/%s: %s", fold_name, split_name, summary["splits"][fold_name][split_name])

    write_json(out_dir / "split_summary.json", summary)
    logger.info("Wrote DOLOS metadata/splits to %s", out_dir)
    logger.info("All label counts: %s", summary["all_label_counts"])


if __name__ == "__main__":
    main()
