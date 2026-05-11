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
from torchvision.models.video import R3D_18_Weights, r3d_18

from src.evaluate import binary_metrics, bootstrap_ci
from src.utils import ensure_dir, set_seed, write_json


KINETICS_MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
KINETICS_STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)


class VideoClipDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        face_metadata_csv: str | Path,
        clip_length: int,
        image_size: int,
        train: bool,
        horizontal_flip: bool = True,
    ) -> None:
        split_df = pd.read_csv(split_csv)
        face_df = pd.read_csv(face_metadata_csv)
        df = split_df.merge(
            face_df[["video_id", "face_video_path", "status", "total_frames"]],
            on="video_id",
            how="left",
        )
        missing = df[df["face_video_path"].isna() | (df["status"] != "ok")]
        if not missing.empty:
            videos = ", ".join(missing["video_id"].astype(str).tolist())
            raise ValueError(f"Missing usable face crops for videos: {videos}")
        self.df = df.reset_index(drop=True)
        self.clip_length = clip_length
        self.image_size = image_size
        self.train = train
        self.horizontal_flip = horizontal_flip

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        total_frames = int(row.get("total_frames", 0) or 0)
        indices = self._sample_indices(total_frames)
        clip = self._read_clip(Path(row["face_video_path"]), indices)
        return {
            "clip": clip,
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
            "video_id": str(row["video_id"]),
            "group_id": str(row.get("group_id", "")),
        }

    def _sample_indices(self, total_frames: int) -> np.ndarray:
        total_frames = max(total_frames, 1)
        if self.train and total_frames > self.clip_length:
            start = np.random.randint(0, total_frames - self.clip_length + 1)
            return np.arange(start, start + self.clip_length)
        return np.linspace(0, total_frames - 1, self.clip_length).round().astype(int)

    def _read_clip(self, video_path: Path, indices: np.ndarray) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        frames: list[torch.Tensor] = []
        last_frame: np.ndarray | None = None
        flip = self.train and self.horizontal_flip and np.random.random() < 0.5
        crop_params = self._crop_params()
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if ok:
                last_frame = frame
            elif last_frame is not None:
                frame = last_frame
            else:
                frame = np.zeros((128, 128, 3), dtype=np.uint8)
            frames.append(self._transform_frame(frame, crop_params, flip))
        cap.release()
        clip = torch.stack(frames, dim=1)
        return (clip - KINETICS_MEAN) / KINETICS_STD

    def _crop_params(self) -> tuple[int, int]:
        max_offset = 128 - self.image_size
        if self.train and max_offset > 0:
            return int(np.random.randint(0, max_offset + 1)), int(np.random.randint(0, max_offset + 1))
        return max(max_offset // 2, 0), max(max_offset // 2, 0)

    def _transform_frame(self, frame_bgr: np.ndarray, crop_params: tuple[int, int], flip: bool) -> torch.Tensor:
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        top, left = crop_params
        frame = frame[top : top + self.image_size, left : left + self.image_size]
        if flip:
            frame = np.ascontiguousarray(frame[:, ::-1])
        return torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0


def load_model(pretrained: bool) -> nn.Module:
    weights = R3D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = trainable or name.startswith("fc.")


def make_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(param)
        else:
            backbone_params.append(param)
    param_groups: list[dict[str, Any]] = [
        {"params": head_params, "lr": float(cfg["training"]["learning_rate_head"])},
    ]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": float(cfg["training"]["learning_rate_backbone"])})
    return torch.optim.AdamW(param_groups, weight_decay=float(cfg["training"]["weight_decay"]))


def make_loader(split_csv: str | Path, face_metadata_csv: str | Path, cfg: dict[str, Any], train: bool, seed: int) -> DataLoader:
    dataset = VideoClipDataset(
        split_csv,
        face_metadata_csv,
        clip_length=int(cfg["data"]["clip_length"]),
        image_size=int(cfg["data"]["image_size"]),
        train=train,
        horizontal_flip=bool(cfg["data"].get("horizontal_flip", True)),
    )
    generator = torch.Generator().manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        np.random.seed(seed + worker_id)

    return DataLoader(
        dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=train,
        num_workers=int(cfg["training"].get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def append_log(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_epoch(
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
        clips = batch["clip"].to(device)
        labels = batch["label"].to(device)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            loss = F.cross_entropy(model(clips), labels) / accumulation_steps
        scaler.scale(loss).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * accumulation_steps * labels.numel()
        total_items += labels.numel()
    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        clips = batch["clip"].to(device)
        labels = batch["label"].numpy()
        scores = torch.softmax(model(clips), dim=1)[:, 1].detach().cpu().numpy()
        for video_id, group_id, label, score in zip(batch["video_id"], batch["group_id"], labels, scores):
            rows.append({"video_id": video_id, "group_id": group_id, "label": int(label), "score_lie": float(score)})
    return pd.DataFrame(rows)


def run(cfg: dict[str, Any], config_path: Path) -> dict[str, Any]:
    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = make_loader(cfg["data"]["train_csv"], cfg["data"]["face_metadata_csv"], cfg, True, seed)
    val_loader = make_loader(cfg["data"]["val_csv"], cfg["data"]["face_metadata_csv"], cfg, False, seed)
    test_loader = make_loader(cfg["data"]["test_csv"], cfg["data"]["face_metadata_csv"], cfg, False, seed)

    model = load_model(bool(cfg["model"].get("pretrained", True))).to(device)
    set_backbone_trainable(model, False)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(cfg["training"].get("scheduler_patience", 3)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    checkpoint_dir = ensure_dir(cfg["outputs"]["checkpoint_dir"])
    metrics_dir = ensure_dir(cfg["outputs"]["metrics_dir"])
    log_path = metrics_dir / "train_log.csv"
    best_path = checkpoint_dir / "best.pt"
    best_auc = -1.0
    best_epoch = -1
    stale = 0
    patience = int(cfg["training"].get("early_stopping_patience", 5))

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        if epoch == int(cfg["training"].get("freeze_backbone_epochs", 0)) + 1:
            set_backbone_trainable(model, True)
            optimizer = make_optimizer(model, cfg)
        loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            int(cfg["training"].get("gradient_accumulation_steps", 1)),
        )
        val_pred = predict(model, val_loader, device)
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
            "head_lr": optimizer.param_groups[0]["lr"],
            "backbone_lr": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else 0.0,
        }
        append_log(log_path, row)
        print(row)
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": cfg,
                    "config_path": str(config_path),
                },
                best_path,
            )
            val_pred.to_csv(metrics_dir / "val_predictions.csv", index=False)
            write_json(metrics_dir / "val_metrics.json", val_metrics)
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch} val_auc_roc={best_auc:.4f}")
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_pred = predict(model, test_loader, device)
    test_metrics = binary_metrics(test_pred["label"].to_numpy(), test_pred["score_lie"].to_numpy())
    test_metrics["bootstrap_95ci"] = bootstrap_ci(test_pred["label"].to_numpy(), test_pred["score_lie"].to_numpy(), seed=seed)
    test_pred.to_csv(metrics_dir / "test_predictions.csv", index=False)
    write_json(metrics_dir / "test_metrics.json", test_metrics)
    summary = {
        "experiment": cfg["experiment"]["name"],
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
    parser = argparse.ArgumentParser(description="Train RGB 3D-CNN baseline.")
    parser.add_argument("--config", default="configs/rgb_3dcnn.yaml")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    summary = run(cfg, config_path)
    print(summary)


if __name__ == "__main__":
    main()
