from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write split CSVs containing only clips with usable multimodal caches.")
    parser.add_argument("--split-dir", default="data/processed/splits/dolos")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dataset", default="dolos")
    parser.add_argument("--audio-root", default="data/processed/audio")
    parser.add_argument("--faces-root", default="data/processed/faces_224_clean")
    parser.add_argument("--optflow-root", default="data/processed/optflow_clean")
    parser.add_argument("--face-manifest", default=None)
    parser.add_argument("--optflow-manifest", default=None)
    parser.add_argument("--validate-npz", action="store_true")
    return parser.parse_args()


def split_files(split_dir: Path) -> list[Path]:
    return sorted(path for path in split_dir.glob("fold*_*.csv") if path.stem.rsplit("_", 1)[-1] in {"train", "val", "test"})


def load_manifest(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} manifest not found: {path}")
    df = pd.read_csv(path)
    if "video_id" not in df.columns or "status" not in df.columns:
        raise ValueError(f"{name} manifest must contain video_id and status columns: {path}")
    return df.drop_duplicates("video_id", keep="last").copy()


def valid_face_ids(face_manifest: pd.DataFrame) -> set[str]:
    valid = face_manifest[face_manifest["status"].isin(["ok", "exists"])].copy()
    if "frames_dir" in valid.columns:
        valid = valid[valid["frames_dir"].map(lambda value: Path(str(value)).exists())].copy()
    return set(valid["video_id"].astype(str))


def valid_flow_ids(optflow_manifest: pd.DataFrame, validate_npz: bool) -> set[str]:
    valid = optflow_manifest[optflow_manifest["status"].isin(["ok", "exists"])].copy()
    if "flow_path" not in valid.columns:
        raise ValueError("optflow manifest must contain flow_path")
    ids: set[str] = set()
    for row in valid.itertuples(index=False):
        video_id = str(getattr(row, "video_id"))
        flow_path = Path(str(getattr(row, "flow_path")))
        if not flow_path.exists():
            continue
        if validate_npz:
            try:
                with np.load(flow_path) as data:
                    flow = data["flow"]
                    if flow.ndim != 4 or flow.shape[1] != 2 or flow.shape[0] < 1:
                        continue
            except Exception:
                continue
        ids.add(video_id)
    return ids


def audio_ids(video_ids: set[str], audio_dir: Path) -> set[str]:
    return {video_id for video_id in video_ids if (audio_dir / f"{video_id}.wav").exists()}


def exclusion_reasons(video_ids: set[str], face_ids: set[str], flow_ids: set[str], wav_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for video_id in sorted(video_ids):
        reasons = []
        if video_id not in face_ids:
            reasons.append("face")
        if video_id not in flow_ids:
            reasons.append("optflow")
        if video_id not in wav_ids:
            reasons.append("audio")
        if reasons:
            rows.append({"video_id": video_id, "missing_or_invalid": ",".join(reasons)})
    return pd.DataFrame(rows)


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).to_dict().items()}


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir)
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else split_dir / "cache_filtered")
    dataset = str(args.dataset)
    audio_dir = Path(args.audio_root) / dataset
    face_manifest_path = Path(args.face_manifest) if args.face_manifest else Path(args.faces_root) / dataset / "preprocess_manifest.csv"
    optflow_manifest_path = Path(args.optflow_manifest) if args.optflow_manifest else Path(args.optflow_root) / dataset / "optflow_manifest.csv"

    files = split_files(split_dir)
    if not files:
        raise FileNotFoundError(f"No fold split CSVs found in {split_dir}")

    split_frames = {path.name: pd.read_csv(path) for path in files}
    all_split_ids = {str(video_id) for df in split_frames.values() for video_id in df["video_id"].astype(str)}
    face_ids = valid_face_ids(load_manifest(face_manifest_path, "face"))
    flow_ids = valid_flow_ids(load_manifest(optflow_manifest_path, "optflow"), args.validate_npz)
    wav_ids = audio_ids(all_split_ids, audio_dir)
    valid_ids = all_split_ids & face_ids & flow_ids & wav_ids

    exclusions = exclusion_reasons(all_split_ids, face_ids, flow_ids, wav_ids)
    exclusions.to_csv(out_dir / "cache_exclusions.csv", index=False)

    summary: dict[str, Any] = {
        "dataset": dataset,
        "source_split_dir": str(split_dir),
        "out_dir": str(out_dir),
        "audio_dir": str(audio_dir),
        "face_manifest": str(face_manifest_path),
        "optflow_manifest": str(optflow_manifest_path),
        "validate_npz": bool(args.validate_npz),
        "input_unique_video_ids": int(len(all_split_ids)),
        "valid_unique_video_ids": int(len(valid_ids)),
        "excluded_unique_video_ids": int(len(all_split_ids - valid_ids)),
        "split_files": {},
    }

    for name, df in split_frames.items():
        before = len(df)
        filtered = df[df["video_id"].astype(str).isin(valid_ids)].copy()
        filtered.to_csv(out_dir / name, index=False)
        summary["split_files"][name] = {
            "input_rows": int(before),
            "output_rows": int(len(filtered)),
            "removed_rows": int(before - len(filtered)),
            "label_counts": label_counts(filtered),
        }

    metadata_path = split_dir / "metadata.csv"
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        metadata = metadata[metadata["video_id"].astype(str).isin(valid_ids)].copy()
        metadata.to_csv(out_dir / "metadata.csv", index=False)
        summary["metadata_rows"] = int(len(metadata))
        summary["metadata_label_counts"] = label_counts(metadata)

    write_json(out_dir / "cache_filter_summary.json", summary)
    print(f"Wrote cache-filtered splits to {out_dir}")
    print(f"Valid split video_ids: {len(valid_ids)}/{len(all_split_ids)}")


if __name__ == "__main__":
    main()
