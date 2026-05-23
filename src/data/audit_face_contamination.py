from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import AutoModel

from src.data.multimodal_dataset import IMAGENET_MEAN, IMAGENET_STD
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit possible face identity contamination in cached face crops.")
    parser.add_argument("--predictions-dir", default="outputs/metrics/dolos_three_stream_gated_logits_prior_kl")
    parser.add_argument("--split-dir", default="data/processed/splits/dolos/cache_filtered")
    parser.add_argument("--faces-root", default="data/processed/faces_224_clean")
    parser.add_argument("--dataset", default="dolos")
    parser.add_argument("--out-dir", default="outputs/metrics/face_contamination_audit")
    parser.add_argument("--model-name", default="LaurenGurgiolo/vit-micro-facial-expressions")
    parser.add_argument("--folds", nargs="+", default=["fold1", "fold2", "fold3"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--sliding-stride-seconds", type=float, default=1.0)
    parser.add_argument("--max-windows-per-clip", type=int, default=16)
    parser.add_argument("--distance-threshold", type=float, default=0.35)
    parser.add_argument("--min-dominant-ratio", type=float, default=0.75)
    parser.add_argument("--limit-clips", type=int, default=None)
    return parser.parse_args()


def sample_frame_indices(
    num_frames: int,
    start_time: float,
    target_fps: float,
    window_seconds: float,
    frames_per_clip: int,
) -> np.ndarray:
    if num_frames <= 1:
        return np.zeros(frames_per_clip, dtype=np.int64)
    duration = num_frames / target_fps
    if duration <= window_seconds:
        start_frame = 0
        end_frame = num_frames - 1
    else:
        start_frame = int(round(start_time * target_fps))
        end_frame = int(round((start_time + window_seconds) * target_fps)) - 1
        start_frame = int(np.clip(start_frame, 0, num_frames - 1))
        end_frame = int(np.clip(end_frame, start_frame, num_frames - 1))
    return np.linspace(start_frame, end_frame, frames_per_clip).round().astype(np.int64)


def load_image_tensor(path: Path, image_size: int = 224) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] != (image_size, image_size):
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def embed_paths(paths: list[Path], model: torch.nn.Module, device: torch.device, batch_size: int) -> dict[Path, np.ndarray]:
    embeddings: dict[Path, np.ndarray] = {}
    model.eval()
    for start in tqdm(range(0, len(paths), batch_size), desc="Embedding face crops"):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([load_image_tensor(path) for path in batch_paths], dim=0).to(device)
        outputs = model(pixel_values=batch)
        cls = outputs.last_hidden_state[:, 0].float()
        cls = torch.nn.functional.normalize(cls, dim=1)
        values = cls.cpu().numpy()
        for path, embedding in zip(batch_paths, values, strict=True):
            embeddings[path] = embedding
    return embeddings


def cluster_window(embeddings: np.ndarray, distance_threshold: float, min_dominant_ratio: float) -> dict[str, Any]:
    if embeddings.shape[0] <= 1:
        return {
            "identity_clusters": 1,
            "dominant_ratio": 1.0,
            "mean_cosine_distance": 0.0,
            "max_cosine_distance": 0.0,
            "suspect_contamination": False,
        }
    similarity = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distances = 1.0 - similarity
    upper = distances[np.triu_indices_from(distances, k=1)]
    max_distance = float(np.max(upper)) if upper.size else 0.0
    mean_distance = float(np.mean(upper)) if upper.size else 0.0
    kwargs: dict[str, Any] = {
        "n_clusters": None,
        "distance_threshold": distance_threshold,
        "linkage": "average",
    }
    try:
        clustering = AgglomerativeClustering(metric="precomputed", **kwargs)
    except TypeError:
        clustering = AgglomerativeClustering(affinity="precomputed", **kwargs)
    labels = clustering.fit_predict(distances)
    counts = np.bincount(labels)
    clusters = int(len(counts))
    dominant_ratio = float(counts.max() / counts.sum())
    suspect = bool(clusters > 1 and dominant_ratio < min_dominant_ratio)
    return {
        "identity_clusters": clusters,
        "dominant_ratio": dominant_ratio,
        "mean_cosine_distance": mean_distance,
        "max_cosine_distance": max_distance,
        "suspect_contamination": suspect,
    }


def binary_error(label: int, score_lie: float, threshold: float) -> int:
    return int((score_lie >= threshold) != bool(label))


def load_fold_data(predictions_dir: Path, split_dir: Path, folds: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    clip_rows: list[pd.DataFrame] = []
    window_rows: list[pd.DataFrame] = []
    for fold in folds:
        split = pd.read_csv(split_dir / f"{fold}_test.csv")
        meta_cols = [col for col in ["video_id", "host", "episode", "group_id", "fold"] if col in split.columns]
        clip_pred = pd.read_csv(predictions_dir / f"{fold}_seed{seed}_test_predictions.csv")
        window_pred = pd.read_csv(predictions_dir / f"{fold}_seed{seed}_test_window_predictions.csv")
        clip_pred = clip_pred.merge(split[meta_cols], on="video_id", how="left")
        window_pred = window_pred.merge(split[meta_cols], on="video_id", how="left")
        clip_pred["fold"] = fold
        window_pred["fold"] = fold
        with (predictions_dir / f"{fold}_seed{seed}_summary.json").open("r", encoding="utf-8") as f:
            summary = json.load(f)
        threshold = float(summary["threshold_calibration"]["threshold"])
        clip_pred["calibrated_threshold"] = threshold
        window_pred["calibrated_threshold"] = threshold
        clip_rows.append(clip_pred)
        window_rows.append(window_pred)
    return pd.concat(clip_rows, ignore_index=True), pd.concat(window_rows, ignore_index=True)


def summarize_error_rate(df: pd.DataFrame, group_col: str, prefix: str) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False):
        rows.append(
            {
                group_col: value,
                f"{prefix}_n": int(len(group)),
                f"{prefix}_default_error_rate": float(group["default_error"].mean()),
                f"{prefix}_calibrated_error_rate": float(group["calibrated_error"].mean()),
                f"{prefix}_mean_score_lie": float(group["score_lie"].mean()),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    render = df.reset_index()
    columns = [str(col) for col in render.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in render.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    predictions_dir = Path(args.predictions_dir)
    split_dir = Path(args.split_dir)
    faces_root = Path(args.faces_root) / args.dataset
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    clip_pred, window_pred = load_fold_data(predictions_dir, split_dir, args.folds, args.seed)
    if args.limit_clips:
        keep_ids = set(clip_pred["video_id"].drop_duplicates().head(args.limit_clips))
        clip_pred = clip_pred[clip_pred["video_id"].isin(keep_ids)].copy()
        window_pred = window_pred[window_pred["video_id"].isin(keep_ids)].copy()

    video_frames: dict[str, list[Path]] = {}
    path_rows: list[dict[str, Any]] = []
    unique_paths: set[Path] = set()
    for video_id, group in tqdm(window_pred.groupby("video_id"), desc="Collecting audit windows"):
        frames = sorted((faces_root / str(video_id)).glob("*.jpg"))
        video_frames[str(video_id)] = frames
        for row in group.itertuples(index=False):
            indices = sample_frame_indices(
                len(frames),
                float(row.start_time),
                args.target_fps,
                args.window_seconds,
                args.frames_per_clip,
            )
            paths = [frames[int(np.clip(idx, 0, len(frames) - 1))] for idx in indices] if frames else []
            unique_paths.update(paths)
            path_rows.append(
                {
                    "fold": row.fold,
                    "video_id": row.video_id,
                    "window_index": int(row.window_index),
                    "start_time": float(row.start_time),
                    "frame_indices": " ".join(str(int(idx)) for idx in indices),
                    "frame_paths": "|".join(str(path) for path in paths),
                    "available_frames": int(len(frames)),
                }
            )

    model = AutoModel.from_pretrained(args.model_name, use_safetensors=True).to(device)
    embeddings = embed_paths(sorted(unique_paths), model, device, args.batch_size)

    audit_rows = []
    for item in tqdm(path_rows, desc="Clustering windows"):
        paths = [Path(value) for value in item["frame_paths"].split("|") if value]
        if paths:
            matrix = np.stack([embeddings[path] for path in paths], axis=0)
            stats = cluster_window(matrix, args.distance_threshold, args.min_dominant_ratio)
        else:
            stats = {
                "identity_clusters": 0,
                "dominant_ratio": 0.0,
                "mean_cosine_distance": 0.0,
                "max_cosine_distance": 0.0,
                "suspect_contamination": True,
            }
        audit_rows.append({**item, **stats})
    window_audit = pd.DataFrame(audit_rows)
    window_audit.to_csv(out_dir / "window_face_contamination_audit.csv", index=False)

    merged_windows = window_pred.merge(
        window_audit[
            [
                "fold",
                "video_id",
                "window_index",
                "identity_clusters",
                "dominant_ratio",
                "mean_cosine_distance",
                "max_cosine_distance",
                "suspect_contamination",
                "available_frames",
            ]
        ],
        on=["fold", "video_id", "window_index"],
        how="left",
    )
    merged_windows["default_error"] = [
        binary_error(int(label), float(score), 0.5) for label, score in zip(merged_windows["label"], merged_windows["score_lie"], strict=True)
    ]
    merged_windows["calibrated_error"] = [
        binary_error(int(label), float(score), float(threshold))
        for label, score, threshold in zip(
            merged_windows["label"],
            merged_windows["score_lie"],
            merged_windows["calibrated_threshold"],
            strict=True,
        )
    ]
    merged_windows.to_csv(out_dir / "window_predictions_with_face_audit.csv", index=False)

    clip_audit = (
        merged_windows.groupby(["fold", "video_id"], as_index=False)
        .agg(
            windows=("window_index", "count"),
            suspect_windows=("suspect_contamination", "sum"),
            first_window_suspect=("suspect_contamination", "first"),
            max_identity_clusters=("identity_clusters", "max"),
            min_dominant_ratio=("dominant_ratio", "min"),
            max_cosine_distance=("max_cosine_distance", "max"),
            mean_cosine_distance=("mean_cosine_distance", "mean"),
        )
    )
    clip_audit["clip_suspect_contamination"] = clip_audit["suspect_windows"] > 0
    clip_audit["suspect_window_ratio"] = clip_audit["suspect_windows"] / clip_audit["windows"].clip(lower=1)
    clip_audit.to_csv(out_dir / "clip_face_contamination_audit.csv", index=False)

    merged_clips = clip_pred.merge(clip_audit, on=["fold", "video_id"], how="left")
    merged_clips["default_error"] = [
        binary_error(int(label), float(score), 0.5) for label, score in zip(merged_clips["label"], merged_clips["score_lie"], strict=True)
    ]
    merged_clips["calibrated_error"] = [
        binary_error(int(label), float(score), float(threshold))
        for label, score, threshold in zip(
            merged_clips["label"],
            merged_clips["score_lie"],
            merged_clips["calibrated_threshold"],
            strict=True,
        )
    ]
    merged_clips.to_csv(out_dir / "clip_predictions_with_face_audit.csv", index=False)

    window_summary = pd.concat(
        [
            summarize_error_rate(merged_windows, "suspect_contamination", "window"),
            summarize_error_rate(merged_windows[merged_windows["window_index"] == 0], "suspect_contamination", "window0"),
        ],
        axis=0,
        ignore_index=False,
    )
    window_summary.to_csv(out_dir / "window_error_by_contamination.csv", index=False)
    clip_summary = pd.concat(
        [
            summarize_error_rate(merged_clips, "clip_suspect_contamination", "clip_any"),
            summarize_error_rate(merged_clips, "first_window_suspect", "clip_first"),
        ],
        axis=0,
        ignore_index=False,
    )
    clip_summary.to_csv(out_dir / "clip_error_by_contamination.csv", index=False)

    top_suspect = merged_clips.sort_values(["suspect_window_ratio", "max_cosine_distance"], ascending=False).head(50)
    top_suspect.to_csv(out_dir / "top_suspect_clips.csv", index=False)
    first_suspect = merged_windows[merged_windows["window_index"] == 0].sort_values("max_cosine_distance", ascending=False).head(50)
    first_suspect.to_csv(out_dir / "top_suspect_window0.csv", index=False)

    lines = [
        "# Face Contamination Audit",
        "",
        f"Predictions: `{predictions_dir}`",
        f"Faces: `{faces_root}`",
        f"Embedding model: `{args.model_name}`",
        f"Distance threshold: `{args.distance_threshold}`; min dominant ratio: `{args.min_dominant_ratio}`",
        "",
        "Important: this is a heuristic audit from cached face crops. It flags likely identity switches by embedding distance/clustering, but it is not manually verified identity ground truth.",
        "",
        "## Counts",
        "",
        f"- Clips audited: {len(merged_clips)}",
        f"- Windows audited: {len(merged_windows)}",
        f"- Suspect clips, any window: {int(merged_clips['clip_suspect_contamination'].sum())} ({merged_clips['clip_suspect_contamination'].mean() * 100:.2f}%)",
        f"- Suspect clips, window 0: {int(merged_clips['first_window_suspect'].sum())} ({merged_clips['first_window_suspect'].mean() * 100:.2f}%)",
        f"- Suspect windows: {int(merged_windows['suspect_contamination'].sum())} ({merged_windows['suspect_contamination'].mean() * 100:.2f}%)",
        f"- Suspect window 0 rows: {int(merged_windows.loc[merged_windows['window_index'] == 0, 'suspect_contamination'].sum())} ({merged_windows.loc[merged_windows['window_index'] == 0, 'suspect_contamination'].mean() * 100:.2f}%)",
        "",
        "## Error Rate By Contamination",
        "",
        "### Clip Any Suspect Window",
        "",
        markdown_table(
            merged_clips.groupby("clip_suspect_contamination").agg(
                n=("video_id", "count"),
                default_error_rate=("default_error", "mean"),
                calibrated_error_rate=("calibrated_error", "mean"),
                mean_score_lie=("score_lie", "mean"),
            )
        ),
        "",
        "### Clip First Window Suspect",
        "",
        markdown_table(
            merged_clips.groupby("first_window_suspect").agg(
                n=("video_id", "count"),
                default_error_rate=("default_error", "mean"),
                calibrated_error_rate=("calibrated_error", "mean"),
                mean_score_lie=("score_lie", "mean"),
            )
        ),
        "",
        "### Window-Level Suspect",
        "",
        markdown_table(
            merged_windows.groupby("suspect_contamination").agg(
                n=("video_id", "count"),
                default_error_rate=("default_error", "mean"),
                calibrated_error_rate=("calibrated_error", "mean"),
                mean_score_lie=("score_lie", "mean"),
            )
        ),
        "",
        "## Outputs",
        "",
        "- `clip_predictions_with_face_audit.csv`",
        "- `window_predictions_with_face_audit.csv`",
        "- `top_suspect_clips.csv`",
        "- `top_suspect_window0.csv`",
    ]
    (out_dir / "face_contamination_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "face_contamination_audit.md")


if __name__ == "__main__":
    main()
