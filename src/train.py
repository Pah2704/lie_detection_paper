from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from src.evaluate import binary_metrics, bootstrap_ci
from src.utils import ensure_dir, set_seed, write_json


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class VideoFrameDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        face_metadata_csv: str | Path,
        frames_per_video: int,
        image_size: int,
        train: bool,
        seed: int,
        horizontal_flip: bool = True,
    ) -> None:
        split_df = pd.read_csv(split_csv)
        face_df = pd.read_csv(face_metadata_csv)
        df = split_df.merge(
            face_df[["video_id", "face_video_path", "status", "total_frames"]],
            on="video_id",
            how="left",
            suffixes=("", "_face"),
        )
        missing = df[df["face_video_path"].isna() | (df["status"] != "ok")]
        if not missing.empty:
            videos = ", ".join(missing["video_id"].astype(str).tolist())
            raise ValueError(f"Missing usable face crops for videos: {videos}")

        self.df = df.reset_index(drop=True)
        self.frames_per_video = frames_per_video
        self.image_size = image_size
        self.train = train
        self.seed = seed
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        video_path = Path(row["face_video_path"])
        total_frames = int(row.get("total_frames", 0) or 0)
        indices = self._sample_indices(idx, total_frames)
        frames = self._read_frames(video_path, indices)
        return {
            "frames": frames,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "video_id": str(row["video_id"]),
            "group_id": str(row.get("group_id", "")),
        }

    def _sample_indices(self, idx: int, total_frames: int) -> np.ndarray:
        if total_frames <= 0:
            total_frames = 1
        if self.train:
            return np.sort(np.random.randint(0, total_frames, size=self.frames_per_video))
        return np.linspace(0, max(total_frames - 1, 0), self.frames_per_video).round().astype(int)

    def _read_frames(self, video_path: Path, indices: np.ndarray) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frames: list[torch.Tensor] = []
        last_frame: np.ndarray | None = None
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok:
                last_frame = frame
            elif last_frame is not None:
                frame = last_frame
            else:
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
            frames.append(self._transform_frame(frame))
        cap.release()
        return torch.stack(frames, dim=0)

    def _transform_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)

        if self.train:
            max_offset = 128 - self.image_size
            top = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
            left = np.random.randint(0, max_offset + 1) if max_offset > 0 else 0
            frame = frame[top : top + self.image_size, left : left + self.image_size]
            if self.horizontal_flip and np.random.random() < 0.5:
                frame = np.ascontiguousarray(frame[:, ::-1])
        else:
            top = max((128 - self.image_size) // 2, 0)
            left = max((128 - self.image_size) // 2, 0)
            frame = frame[top : top + self.image_size, left : left + self.image_size]

        tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def load_model(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def flatten_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    frames = batch["frames"].to(device)
    labels = batch["label"].to(device)
    batch_size, frames_per_video = frames.shape[:2]
    frames = frames.view(batch_size * frames_per_video, *frames.shape[2:])
    labels = labels.repeat_interleave(frames_per_video)
    return frames, labels


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    accumulation_steps: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        frames, labels = flatten_batch(batch, device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(frames)
            loss = F.cross_entropy(logits, labels) / accumulation_steps

        scaler.scale(loss).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().cpu()) * accumulation_steps * labels.numel()
        total_items += labels.numel()

    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict_videos(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        frames = batch["frames"].to(device)
        labels = batch["label"].numpy()
        batch_size, frames_per_video = frames.shape[:2]
        frames = frames.view(batch_size * frames_per_video, *frames.shape[2:])
        logits = model(frames)
        probs = torch.softmax(logits, dim=1)[:, 1].view(batch_size, frames_per_video)
        scores = probs.mean(dim=1).detach().cpu().numpy()

        for video_id, group_id, label, score in zip(batch["video_id"], batch["group_id"], labels, scores):
            rows.append(
                {
                    "video_id": video_id,
                    "group_id": group_id,
                    "label": int(label),
                    "score_lie": float(score),
                }
            )
    return pd.DataFrame(rows)


def make_loader(
    split_csv: str | Path,
    face_metadata_csv: str | Path,
    config: dict[str, Any],
    train: bool,
    seed: int,
) -> DataLoader:
    data_cfg = config["data"]
    train_cfg = config["training"]
    dataset = VideoFrameDataset(
        split_csv=split_csv,
        face_metadata_csv=face_metadata_csv,
        frames_per_video=int(data_cfg["frames_per_video"]),
        image_size=int(data_cfg["image_size"]),
        train=train,
        seed=seed,
        horizontal_flip=bool(data_cfg.get("horizontal_flip", True)),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)

    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=train,
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def append_log(log_path: Path, row: dict[str, Any]) -> None:
    ensure_dir(log_path.parent)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    seed = int(config["experiment"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(config["data"]["train_csv"], config["data"]["face_metadata_csv"], config, True, seed)
    val_loader = make_loader(config["data"]["val_csv"], config["data"]["face_metadata_csv"], config, False, seed)
    test_loader = make_loader(config["data"]["test_csv"], config["data"]["face_metadata_csv"], config, False, seed)

    model = load_model(bool(config["model"].get("pretrained", True))).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(config["training"].get("scheduler_patience", 3)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = ensure_dir(config["outputs"]["checkpoint_dir"])
    metrics_dir = ensure_dir(config["outputs"]["metrics_dir"])
    log_path = metrics_dir / "train_log.csv"
    best_path = out_dir / "best.pt"

    best_auc = -1.0
    best_epoch = -1
    stale_epochs = 0
    patience = int(config["training"].get("early_stopping_patience", 6))

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            accumulation_steps=int(config["training"].get("gradient_accumulation_steps", 1)),
        )
        val_pred = predict_videos(model, val_loader, device)
        val_metrics = binary_metrics(val_pred["label"].to_numpy(), val_pred["score_lie"].to_numpy())
        val_auc = float(val_metrics["auc_roc"])
        scheduler.step(val_auc)

        row = {
            "epoch": epoch,
            "train_loss": loss,
            "val_auc_roc": val_auc,
            "val_auc_pr": val_metrics["auc_pr"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_lie_recall": val_metrics["recall_lie"],
            "val_eer": val_metrics["eer"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_log(log_path, row)
        print(row)

        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                    "config_path": str(config_path),
                },
                best_path,
            )
            val_pred.to_csv(metrics_dir / "val_predictions.csv", index=False)
            write_json(metrics_dir / "val_metrics.json", val_metrics)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} val_auc_roc={best_auc:.4f}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    test_pred = predict_videos(model, test_loader, device)
    test_metrics = binary_metrics(test_pred["label"].to_numpy(), test_pred["score_lie"].to_numpy())
    test_metrics["bootstrap_95ci"] = bootstrap_ci(
        test_pred["label"].to_numpy(),
        test_pred["score_lie"].to_numpy(),
        seed=seed,
    )
    test_pred.to_csv(metrics_dir / "test_predictions.csv", index=False)
    write_json(metrics_dir / "test_metrics.json", test_metrics)

    summary = {
        "experiment": config["experiment"]["name"],
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_auc_roc": best_auc,
        "test_metrics": test_metrics,
        "checkpoint": str(best_path),
    }
    write_json(metrics_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train visual baselines.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--seed", type=int, help="Override experiment seed.")
    parser.add_argument("--experiment-name", help="Override experiment name.")
    parser.add_argument("--checkpoint-dir", help="Override checkpoint output directory.")
    parser.add_argument("--metrics-dir", help="Override metrics output directory.")
    parser.add_argument("--no-pretrained", action="store_true", help="Override config and train ResNet18 from scratch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed
    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name
    if args.checkpoint_dir:
        config["outputs"]["checkpoint_dir"] = args.checkpoint_dir
    if args.metrics_dir:
        config["outputs"]["metrics_dir"] = args.metrics_dir
    if args.no_pretrained:
        config["model"]["pretrained"] = False
    summary = train(config, config_path)
    print(summary)


if __name__ == "__main__":
    main()
