from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train_multimodal import build_model, face_valid_for_model, make_loader, split_csv_from_config
from src.utils import ensure_dir, read_yaml, write_json


STREAMS = ("static", "flow", "audio")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-stream vote disagreement for gated multimodal checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--out-dir", default="outputs/metrics/stream_disagreement")
    parser.add_argument("--folds", default="fold1 fold2 fold3")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--consensus-bonus", type=float, default=0.25)
    return parser.parse_args()


def stream_keys(stream: str) -> str:
    return "spatial" if stream == "static" else stream


def pattern_name(static_vote: int, flow_vote: int, audio_vote: int) -> str:
    votes = {"static": static_vote, "flow": flow_vote, "audio": audio_vote}
    if static_vote == flow_vote == audio_vote:
        return "all_lie" if static_vote == 1 else "all_truth"
    if flow_vote == audio_vote and static_vote != flow_vote:
        return "static_vs_flow_audio"
    if static_vote == audio_vote and flow_vote != static_vote:
        return "flow_vs_static_audio"
    if static_vote == flow_vote and audio_vote != static_vote:
        return "audio_vs_static_flow"
    return "other"


def support_prediction(row: pd.Series, consensus_bonus: float = 0.0) -> float:
    lie_support = 0.0
    truth_support = 0.0
    lie_votes = 0
    truth_votes = 0
    for stream in STREAMS:
        p_lie = float(row[f"{stream}_score_lie"])
        reliability = float(row["face_valid_ratio"]) if stream in {"static", "flow"} else 1.0
        if p_lie >= 0.5:
            lie_support += reliability * p_lie
            lie_votes += 1
        else:
            truth_support += reliability * (1.0 - p_lie)
            truth_votes += 1
    if consensus_bonus > 0.0:
        lie_support *= 1.0 + consensus_bonus * max(lie_votes - 1, 0)
        truth_support *= 1.0 + consensus_bonus * max(truth_votes - 1, 0)
    return lie_support - truth_support


def summarize_pattern(group: pd.DataFrame, threshold: float, consensus_bonus: float) -> dict[str, Any]:
    labels = group["label"].astype(int)
    final_pred = (group["score_lie"] >= threshold).astype(int)
    majority_pred = ((group["static_vote"] + group["flow_vote"] + group["audio_vote"]) >= 2).astype(int)
    support_pred = (group["support_margin"] >= 0.0).astype(int)
    support_bonus_pred = (group["support_bonus_margin"] >= 0.0).astype(int)
    row: dict[str, Any] = {
        "count": int(len(group)),
        "lie_rate": float(labels.mean()) if len(group) else 0.0,
        "final_acc": float((final_pred == labels).mean()) if len(group) else 0.0,
        "majority_acc": float((majority_pred == labels).mean()) if len(group) else 0.0,
        "support_acc": float((support_pred == labels).mean()) if len(group) else 0.0,
        "support_bonus_acc": float((support_bonus_pred == labels).mean()) if len(group) else 0.0,
    }
    for stream in STREAMS:
        row[f"{stream}_acc"] = float((group[f"{stream}_vote"].astype(int) == labels).mean()) if len(group) else 0.0
        row[f"{stream}_mean_score_lie"] = float(group[f"{stream}_score_lie"].mean()) if len(group) else 0.0
    row["mean_face_valid_ratio"] = float(group["face_valid_ratio"].mean()) if len(group) else 0.0
    return row


@torch.no_grad()
def collect_clip_streams(
    config: dict[str, Any],
    checkpoint: Path,
    fold: str,
    seed: int,
    split: str,
    device: torch.device,
) -> pd.DataFrame:
    model = build_model(config).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()

    split_csv = split_csv_from_config(config, split, fold)
    loader = make_loader(config, split_csv, "dolos", train=False, seed=seed)
    amp = bool(config["training"].get("amp", True))
    face_valid_mode = str(config["data"].get("face_valid_mode", "binary"))
    rows: list[dict[str, Any]] = []
    for batch in loader:
        rgb = batch["rgb_frames"].to(device, non_blocking=True)
        flow = batch["optflow_frames"].to(device, non_blocking=True)
        audio = batch["audio_waveform"].to(device, non_blocking=True)
        face_valid = face_valid_for_model(batch, device, face_valid_mode)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits, features = model(rgb, flow, audio, face_valid=face_valid, return_features=True)
            score_lie = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            stream_scores = {}
            for stream in STREAMS:
                logits_key = f"calibrated_logits_{stream_keys(stream)}"
                stream_scores[stream] = F.softmax(features[logits_key], dim=1)[:, 1].detach().cpu()
        labels = batch["label"].cpu()
        face_valid_ratio = batch["face_valid_ratio"].cpu()
        for idx in range(len(labels)):
            rows.append(
                {
                    "video_id": batch["video_id"][idx],
                    "label": int(labels[idx]),
                    "score_lie": float(score_lie[idx]),
                    "static_score_lie": float(stream_scores["static"][idx]),
                    "flow_score_lie": float(stream_scores["flow"][idx]),
                    "audio_score_lie": float(stream_scores["audio"][idx]),
                    "face_valid_ratio": float(face_valid_ratio[idx]),
                }
            )
    windows = pd.DataFrame(rows)
    grouped = (
        windows.groupby("video_id", as_index=False)
        .agg(
            label=("label", "first"),
            score_lie=("score_lie", "mean"),
            static_score_lie=("static_score_lie", "mean"),
            flow_score_lie=("flow_score_lie", "mean"),
            audio_score_lie=("audio_score_lie", "mean"),
            face_valid_ratio=("face_valid_ratio", "mean"),
        )
        .sort_values("video_id")
    )
    for stream in STREAMS:
        grouped[f"{stream}_vote"] = (grouped[f"{stream}_score_lie"] >= 0.5).astype(int)
    grouped["pattern"] = grouped.apply(
        lambda row: pattern_name(int(row["static_vote"]), int(row["flow_vote"]), int(row["audio_vote"])),
        axis=1,
    )
    grouped["support_margin"] = grouped.apply(lambda row: support_prediction(row, 0.0), axis=1)
    grouped["support_bonus_margin"] = grouped.apply(lambda row: support_prediction(row, consensus_bonus=0.25), axis=1)
    grouped.insert(0, "fold", fold)
    grouped.insert(1, "seed", seed)
    grouped.insert(2, "split", split)
    return grouped


def write_markdown(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    lines = [f"# {title}", "", "| Pattern | Count | Lie rate | Final acc | Majority acc | Support acc | Support+bonus acc | Static acc | Flow acc | Audio acc | Face valid |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for _, row in summary.iterrows():
        lines.append(
            "| {pattern} | {count:d} | {lie_rate:.2f} | {final_acc:.2f} | {majority_acc:.2f} | {support_acc:.2f} | {support_bonus_acc:.2f} | {static_acc:.2f} | {flow_acc:.2f} | {audio_acc:.2f} | {mean_face_valid_ratio:.2f} |".format(
                **row.to_dict()
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_dir = Path(args.checkpoint_dir or config["outputs"]["checkpoint_dir"])
    out_dir = ensure_dir(args.out_dir)
    folds = args.folds.split()
    seeds = [int(seed) for seed in args.seeds.split()]
    all_rows = []
    for fold in folds:
        for seed in seeds:
            checkpoint = checkpoint_dir / f"{fold}_seed{seed}_best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            all_rows.append(collect_clip_streams(config, checkpoint, fold, seed, args.split, device))
    clips = pd.concat(all_rows, ignore_index=True)
    clips["support_bonus_margin"] = clips.apply(lambda row: support_prediction(row, args.consensus_bonus), axis=1)
    summary_rows = []
    for pattern, group in clips.groupby("pattern"):
        row = {"pattern": pattern}
        row.update(summarize_pattern(group, args.threshold, args.consensus_bonus))
        summary_rows.append(row)
    overall = {"pattern": "overall"}
    overall.update(summarize_pattern(clips, args.threshold, args.consensus_bonus))
    summary = pd.DataFrame([overall, *summary_rows])
    stem = f"{Path(args.config).stem}_{args.split}_seed{'-'.join(str(seed) for seed in seeds)}"
    clips.to_csv(out_dir / f"{stem}_clip_stream_votes.csv", index=False)
    summary.to_csv(out_dir / f"{stem}_summary.csv", index=False)
    write_json(out_dir / f"{stem}_summary.json", summary.to_dict(orient="records"))
    write_markdown(summary, out_dir / f"{stem}_summary.md", f"Stream Disagreement: {Path(args.config).stem} ({args.split})")


if __name__ == "__main__":
    main()
