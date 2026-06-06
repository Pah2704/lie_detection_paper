from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.multimodal_dataset import MultimodalClipDataset, multimodal_collate
from src.evaluate import aggregate_windows, binary_metrics, bootstrap_ci, find_best_threshold
from src.models.fusion_cross_attention import ThreeStreamModel
from src.models.losses import FocalLoss, build_loss
from src.utils import append_experiment_rows, ensure_dir, log_args, read_yaml, render_experiment_journal, set_seed, setup_logger, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the three-stream multimodal model.")
    parser.add_argument("--config", default="configs/multimodal_dolos.yaml")
    parser.add_argument("--fold", default="fold1")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--journal-root", default="outputs/report_journal")
    parser.add_argument("--journal-tag", default="")
    parser.add_argument("--journal-notes", default="")
    return parser.parse_args()


def split_csv_from_config(config: dict[str, Any], split: str, fold: str) -> str:
    data_cfg = config["data"]
    split_dir = data_cfg.get("split_dir")
    if split_dir:
        return str(Path(split_dir) / f"{fold}_{split}.csv")
    explicit = data_cfg.get(f"{split}_csv")
    if explicit and f"{fold}_{split}" in explicit:
        return explicit
    return f"data/processed/splits/dolos/{fold}_{split}.csv"


def make_loader(config: dict[str, Any], split_csv: str, dataset_name: str, train: bool, seed: int) -> DataLoader:
    data_cfg = config["data"]
    dataset = MultimodalClipDataset(
        split_csv=split_csv,
        dataset_name=dataset_name,
        audio_root=data_cfg["audio_dir"],
        faces_root=data_cfg["faces_dir"],
        face_valid_root=data_cfg.get("face_valid_root"),
        optflow_root=data_cfg["optflow_dir"],
        train=train,
        frames_per_clip=int(data_cfg["frames_per_clip"]),
        window_seconds=float(data_cfg["window_seconds"]),
        num_windows_per_clip_train=int(data_cfg["num_windows_per_clip_train"]),
        sliding_stride_seconds=float(data_cfg["sliding_stride_seconds"]),
        max_windows_per_clip=int(data_cfg["max_windows_per_clip"]),
        min_window_start_seconds=float(data_cfg.get("min_window_start_seconds", 0.0)),
        min_window_face_valid_ratio=float(data_cfg.get("min_window_face_valid_ratio", 0.75)),
        use_face_valid_mask=bool(data_cfg.get("use_face_valid_mask", True)),
        sample_rate=int(data_cfg["sample_rate"]),
        image_size=int(data_cfg["image_size"]),
        speed_perturb_range=tuple(data_cfg.get("speed_perturb_range", [0.9, 1.1])),
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=train,
        num_workers=int(data_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=multimodal_collate,
        generator=generator,
    )


def face_valid_for_model(batch: dict[str, Any], device: torch.device, mode: str) -> torch.Tensor | None:
    normalized = mode.lower().strip()
    if normalized in {"none", "off", "disabled"}:
        return None
    if "face_valid_frames" in batch:
        return batch["face_valid_frames"].to(device, non_blocking=True)
    if normalized in {"ratio", "soft", "face_valid_ratio"}:
        key = "face_valid_ratio"
    elif normalized in {"binary", "hard", "face_valid"}:
        key = "face_valid"
    else:
        raise ValueError(f"Unsupported data.face_valid_mode: {mode}")
    return batch[key].to(device, non_blocking=True)


def build_model(config: dict[str, Any]) -> ThreeStreamModel:
    model_cfg = config["model"]
    return ThreeStreamModel(
        vit_name=model_cfg["vit_name"],
        wav2vec_name=model_cfg["wav2vec_name"],
        projection_dim=int(model_cfg["projection_dim"]),
        attention_heads=int(model_cfg["attention_heads"]),
        lstm_hidden_dim=int(model_cfg["lstm_hidden_dim"]),
        lstm_layers=int(model_cfg["lstm_layers"]),
        freeze_vit=bool(model_cfg.get("freeze_vit", True)),
        freeze_wav2vec=bool(model_cfg.get("freeze_wav2vec", True)),
        stream_mode=str(model_cfg.get("stream_mode", "all")),
        gate_hidden_dim=int(model_cfg.get("gate_hidden_dim", 128)),
        stream_dropout=float(model_cfg.get("stream_dropout", 0.0)),
        gate_init_weights=tuple(float(value) for value in model_cfg.get("gate_init_weights", [])) or None,
        stream_logit_temperature=tuple(float(value) for value in model_cfg.get("stream_logit_temperature", [])) or None,
        logit_fusion_mode=str(model_cfg.get("logit_fusion_mode", "weighted_logits")),
        margin_clip=float(model_cfg["margin_clip"]) if "margin_clip" in model_cfg else None,
        reliability_confidence_gamma=float(model_cfg.get("reliability_confidence_gamma", 1.0)),
        reliability_confidence_floor=float(model_cfg.get("reliability_confidence_floor", 0.0)),
        reliability_confidence_max_weight=(
            None
            if model_cfg.get("reliability_confidence_max_weight") is None
            else float(model_cfg["reliability_confidence_max_weight"])
        ),
        majority_consistency_margin=(
            None if model_cfg.get("majority_consistency_margin") is None else float(model_cfg["majority_consistency_margin"])
        ),
        vote_consensus_bonus=float(model_cfg.get("vote_consensus_bonus", 0.0)),
        audio_reliability=float(model_cfg.get("audio_reliability", 1.0)),
        confidence_detach=bool(model_cfg.get("confidence_detach", True)),
        use_temporal_positional_encoding=bool(model_cfg.get("use_temporal_positional_encoding", False)),
        local_attention_radius=(
            int(model_cfg["local_attention_radius"]) if model_cfg.get("local_attention_radius") is not None else None
        ),
        temporal_position_scale=float(model_cfg.get("temporal_position_scale", 1.0)),
        temporal_attention_sigma=(
            float(model_cfg["temporal_attention_sigma"]) if model_cfg.get("temporal_attention_sigma") is not None else None
        ),
    )


def choose_loss(config: dict[str, Any], train_csv: str) -> tuple[torch.nn.Module, str, float]:
    labels = torch.tensor(pd.read_csv(train_csv)["label"].astype(int).to_numpy(), dtype=torch.long)
    counts = torch.bincount(labels, minlength=2).float()
    imbalance = float((counts.max() - counts.min()) / counts.sum().clamp_min(1.0))
    loss_name = str(config["training"].get("loss", "auto")).lower()
    if loss_name == "auto":
        loss_name = "ce" if imbalance < 0.10 else "focal"
    criterion = build_loss(loss_name, labels=labels, gamma=float(config["training"].get("focal_loss_gamma", 2.0)))
    return criterion, loss_name, imbalance


def append_log(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def gate_entropy(gate_weights: torch.Tensor) -> torch.Tensor:
    return -(gate_weights.clamp_min(1e-8).log() * gate_weights).sum(dim=1).mean()


def normalized_prior_tensor(
    weights: tuple[float, ...],
    gate_weights: torch.Tensor,
) -> torch.Tensor:
    prior = torch.tensor(weights, dtype=gate_weights.dtype, device=gate_weights.device)
    if prior.numel() != gate_weights.size(1) or torch.any(prior <= 0):
        raise ValueError(f"Gate prior must contain {gate_weights.size(1)} positive values.")
    return prior / prior.sum().clamp_min(1e-8)


def adaptive_gate_prior(
    gate_weights: torch.Tensor,
    face_valid: torch.Tensor | None,
    low_face_weights: tuple[float, ...],
    high_face_weights: tuple[float, ...],
    face_power: float = 1.0,
) -> torch.Tensor:
    low = normalized_prior_tensor(low_face_weights, gate_weights)
    high = normalized_prior_tensor(high_face_weights, gate_weights)
    if face_valid is None:
        strength = torch.full((gate_weights.size(0), 1), 0.5, dtype=gate_weights.dtype, device=gate_weights.device)
    else:
        strength = face_valid.to(gate_weights.device, gate_weights.dtype).view(-1, 1).clamp(0.0, 1.0)
    if face_power <= 0.0:
        raise ValueError("training.gate_prior_face_power must be positive.")
    strength = strength.pow(face_power)
    prior = low.view(1, -1) + strength * (high.view(1, -1) - low.view(1, -1))
    return prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)


def gate_prior_kl(
    gate_weights: torch.Tensor,
    prior_weights: tuple[float, ...] | torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if isinstance(prior_weights, torch.Tensor):
        prior = prior_weights.to(device=gate_weights.device, dtype=gate_weights.dtype)
        if prior.dim() == 1:
            prior = prior.view(1, -1)
        if prior.shape[-1] != gate_weights.size(1):
            raise ValueError(f"Gate prior must have {gate_weights.size(1)} streams, got {prior.shape[-1]}.")
        if prior.size(0) not in {1, gate_weights.size(0)} or torch.any(prior <= 0):
            raise ValueError("Gate prior tensor must be [streams] or [batch, streams] with positive values.")
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1e-8)
    else:
        prior = normalized_prior_tensor(prior_weights, gate_weights).view(1, -1)
    per_sample = (gate_weights * (gate_weights.clamp_min(1e-8).log() - prior.clamp_min(1e-8).log())).sum(dim=1)
    if sample_weight is None:
        return per_sample.mean()
    weights = sample_weight.to(per_sample.device, per_sample.dtype).view(-1).clamp(0.0, 1.0)
    if float(weights.sum().detach().cpu()) <= 0.0:
        return per_sample.sum() * 0.0
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def criterion_with_sample_weight(
    criterion: torch.nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if sample_weight is None:
        return criterion(logits, labels)
    weights = sample_weight.to(logits.device, logits.dtype).view(-1).clamp(0.0, 1.0)
    if float(weights.sum().detach().cpu()) <= 0.0:
        return logits.sum() * 0.0
    if isinstance(criterion, torch.nn.CrossEntropyLoss):
        per_sample = F.cross_entropy(
            logits,
            labels,
            weight=criterion.weight,
            ignore_index=criterion.ignore_index,
            reduction="none",
            label_smoothing=criterion.label_smoothing,
        )
    elif isinstance(criterion, FocalLoss):
        ce = F.cross_entropy(logits, labels, reduction="none", weight=criterion.alpha)
        per_sample = (1.0 - torch.exp(-ce)) ** criterion.gamma * ce
    else:
        return criterion(logits, labels)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def training_loss_from_output(
    output: torch.Tensor | dict[str, torch.Tensor],
    labels: torch.Tensor,
    criterion: torch.nn.Module,
    aux_loss_weight: float,
    gate_entropy_weight: float,
    gate_prior_weight: float = 0.0,
    gate_prior_weights: tuple[float, ...] = (),
    aux_loss_weights: tuple[float, float, float] | None = None,
    gate_prior_mode: str = "fixed",
    gate_prior_low_face_weights: tuple[float, ...] = (),
    gate_prior_high_face_weights: tuple[float, ...] = (),
    gate_prior_face_power: float = 1.0,
    gate_prior_sample_weight: str = "face_valid",
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return criterion(output, labels), output

    logits = output["logits"]
    loss = criterion(logits, labels)
    face_valid = output.get("face_valid")
    stream_aux_weights = aux_loss_weights or (aux_loss_weight, aux_loss_weight, aux_loss_weight)
    for key, weight in zip(("logits_spatial", "logits_flow", "logits_audio"), stream_aux_weights, strict=True):
        if weight > 0.0 and key in output:
            sample_weight = face_valid if key in {"logits_spatial", "logits_flow"} else None
            loss = loss + weight * criterion_with_sample_weight(criterion, output[key], labels, sample_weight)
    if gate_entropy_weight != 0.0 and "gate_weights" in output:
        # Positive weights maximize entropy, discouraging hard early stream collapse.
        loss = loss - gate_entropy_weight * gate_entropy(output["gate_weights"])
    if gate_prior_weight > 0.0 and "gate_weights" in output:
        mode = gate_prior_mode.lower().strip()
        sample_weight_mode = gate_prior_sample_weight.lower().strip()
        sample_weight = face_valid if sample_weight_mode in {"face_valid", "face-valid", "valid"} else None
        if sample_weight_mode not in {"none", "off", "disabled", "face_valid", "face-valid", "valid"}:
            raise ValueError(f"Unsupported training.gate_prior_sample_weight: {gate_prior_sample_weight}")
        if mode in {"fixed", "static"}:
            if not gate_prior_weights:
                raise ValueError("training.gate_prior_weights is required when gate_prior_mode=fixed.")
            prior = gate_prior_weights
        elif mode in {"face_valid_adaptive", "adaptive", "face-valid-adaptive"}:
            if not gate_prior_low_face_weights or not gate_prior_high_face_weights:
                raise ValueError(
                    "training.gate_prior_low_face_weights and gate_prior_high_face_weights are required "
                    "when gate_prior_mode=face_valid_adaptive."
                )
            prior = adaptive_gate_prior(
                output["gate_weights"],
                face_valid,
                gate_prior_low_face_weights,
                gate_prior_high_face_weights,
                gate_prior_face_power,
            )
        else:
            raise ValueError(f"Unsupported training.gate_prior_mode: {gate_prior_mode}")
        loss = loss + gate_prior_weight * gate_prior_kl(output["gate_weights"], prior, sample_weight)
    return loss, logits


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    accumulation_steps: int,
    amp: bool,
    aux_loss_weight: float = 0.0,
    gate_entropy_weight: float = 0.0,
    gate_prior_weight: float = 0.0,
    gate_prior_weights: tuple[float, ...] = (),
    aux_loss_weights: tuple[float, float, float] | None = None,
    face_valid_mode: str = "binary",
    gate_prior_mode: str = "fixed",
    gate_prior_low_face_weights: tuple[float, ...] = (),
    gate_prior_high_face_weights: tuple[float, ...] = (),
    gate_prior_face_power: float = 1.0,
    gate_prior_sample_weight: str = "face_valid",
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, start=1):
        rgb = batch["rgb_frames"].to(device, non_blocking=True)
        flow = batch["optflow_frames"].to(device, non_blocking=True)
        audio = batch["audio_waveform"].to(device, non_blocking=True)
        face_valid = face_valid_for_model(batch, device, face_valid_mode)
        labels = batch["label"].to(device, non_blocking=True)
        aux_loss_active = any(weight > 0.0 for weight in (aux_loss_weights or (aux_loss_weight, aux_loss_weight, aux_loss_weight)))
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            output = model(
                rgb,
                flow,
                audio,
                face_valid=face_valid,
                return_aux=aux_loss_active or gate_entropy_weight != 0.0 or gate_prior_weight > 0.0,
            )
            loss, _ = training_loss_from_output(
                output,
                labels,
                criterion,
                aux_loss_weight,
                gate_entropy_weight,
                gate_prior_weight,
                gate_prior_weights,
                aux_loss_weights,
                gate_prior_mode,
                gate_prior_low_face_weights,
                gate_prior_high_face_weights,
                gate_prior_face_power,
                gate_prior_sample_weight,
            )
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        if step % accumulation_steps == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * accumulation_steps * labels.numel()
        total_items += labels.numel()
    return total_loss / max(total_items, 1)


@torch.no_grad()
def predict_windows(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    face_valid_mode: str = "binary",
) -> pd.DataFrame:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        rgb = batch["rgb_frames"].to(device, non_blocking=True)
        flow = batch["optflow_frames"].to(device, non_blocking=True)
        audio = batch["audio_waveform"].to(device, non_blocking=True)
        face_valid = face_valid_for_model(batch, device, face_valid_mode)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            logits = model(rgb, flow, audio, face_valid=face_valid)
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labels = batch["label"].cpu().numpy()
        face_valid_values = batch["face_valid"].cpu().numpy()
        face_valid_ratios = batch["face_valid_ratio"].cpu().numpy()
        for i, score in enumerate(scores):
            rows.append(
                {
                    "video_id": batch["video_id"][i],
                    "label": int(labels[i]),
                    "score_lie": float(score),
                    "window_index": int(batch["window_index"][i]),
                    "num_windows": int(batch["num_windows"][i]),
                    "start_time": float(batch["start_time"][i]),
                    "face_valid": int(face_valid_values[i] >= 0.5),
                    "face_valid_ratio": float(face_valid_ratios[i]),
                }
            )
    return pd.DataFrame(rows)


def mean_metrics(window_predictions: pd.DataFrame) -> dict[str, Any]:
    mean_pred = aggregate_windows(window_predictions)["mean"]
    return binary_metrics(mean_pred["label"].to_numpy(), mean_pred["score_lie"].to_numpy())


def normalize_metric_name(metric: str) -> str:
    normalized = metric.lower()
    if normalized.startswith("val_"):
        normalized = normalized[4:]
    aliases = {
        "balanced_acc": "balanced_accuracy",
        "bal_acc": "balanced_accuracy",
        "f1": "f1_lie",
    }
    return aliases.get(normalized, normalized)


def metric_value(metrics: dict[str, Any], metric: str) -> tuple[str, float]:
    key = normalize_metric_name(metric)
    if key not in metrics:
        available = ", ".join(sorted(k for k, value in metrics.items() if isinstance(value, int | float)))
        raise ValueError(f"Metric '{metric}' is not available. Available scalar metrics: {available}")
    value = float(metrics[key])
    if value != value:
        return key, -1.0
    return key, value


def confusion_columns(metrics: dict[str, Any], prefix: str) -> dict[str, int]:
    matrix = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    return {
        f"{prefix}_tn": int(matrix[0][0]),
        f"{prefix}_fp": int(matrix[0][1]),
        f"{prefix}_fn": int(matrix[1][0]),
        f"{prefix}_tp": int(matrix[1][1]),
    }


def tensor_norm_columns(name: str, tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float()
    return {
        f"feature_{name}_abs_mean": float(values.abs().mean().cpu()),
        f"feature_{name}_rms": float(torch.sqrt(torch.mean(values.square())).cpu()),
        f"feature_{name}_std": float(values.std(unbiased=False).cpu()),
        f"feature_{name}_max_abs": float(values.abs().max().cpu()),
    }


@torch.no_grad()
def collect_feature_norms(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_batches: int = 1,
    face_valid_mode: str = "binary",
) -> dict[str, float]:
    model.eval()
    collected: dict[str, list[float]] = {}
    for batch_idx, batch in enumerate(loader, start=1):
        rgb = batch["rgb_frames"].to(device, non_blocking=True)
        flow = batch["optflow_frames"].to(device, non_blocking=True)
        audio = batch["audio_waveform"].to(device, non_blocking=True)
        face_valid = face_valid_for_model(batch, device, face_valid_mode)
        with torch.amp.autocast("cuda", enabled=amp and device.type == "cuda"):
            _, features = model(rgb, flow, audio, face_valid=face_valid, return_features=True)
        for name in sorted(features):
            for key, value in tensor_norm_columns(name, features[name]).items():
                collected.setdefault(key, []).append(value)
        if batch_idx >= max_batches:
            break
    return {key: float(sum(values) / len(values)) for key, values in collected.items() if values}


def maybe_init_wandb(config: dict[str, Any], disabled: bool, logger) -> Any:
    if disabled:
        logger.info("W&B disabled by flag.")
        return None
    try:
        import wandb

        run = wandb.init(
            project=config["outputs"].get("wandb_project", "lie-nonlie-multimodal"),
            name=config["experiment"]["name"],
            config=config,
        )
        logger.info("W&B initialized: %s", run.name)
        return run
    except Exception as exc:  # noqa: BLE001 - logging initialization failure should not stop training.
        logger.warning("W&B init failed: %s", exc)
        return None


def append_train_journal(
    journal_root: str | Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    summary: dict[str, Any],
    loss_name: str,
    imbalance: float,
) -> tuple[Path, Path]:
    journal_root = ensure_dir(journal_root)
    csv_path = journal_root / "experiment_journal.csv"
    md_path = journal_root / "experiment_journal.md"
    test_metrics = summary["test_metrics"]
    row = {
        "recorded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": "train",
        "experiment_name": config["experiment"]["name"],
        "tag": args.journal_tag,
        "dataset": "dolos",
        "fold": summary["fold"],
        "seed": summary["seed"],
        "aggregation": "mean",
        "loss_name": loss_name,
        "label_imbalance": imbalance,
        "batch_size": int(config["training"]["batch_size"]),
        "gradient_accumulation": int(config["training"]["gradient_accumulation"]),
        "frames_per_clip": int(config["data"]["frames_per_clip"]),
        "window_seconds": float(config["data"]["window_seconds"]),
        "num_windows_per_clip_train": int(config["data"]["num_windows_per_clip_train"]),
        "freeze_vit": bool(config["model"].get("freeze_vit", True)),
        "freeze_wav2vec": bool(config["model"].get("freeze_wav2vec", True)),
        "stream_mode": str(config["model"].get("stream_mode", "all")),
        "best_epoch": summary["best_epoch"],
        "best_val_f1_lie": summary["best_val_f1_lie"],
        "best_val_metric_name": summary.get("best_val_metric_name"),
        "best_val_metric": summary.get("best_val_metric"),
        "threshold_calibration_metric": summary.get("threshold_calibration", {}).get("metric"),
        "threshold": test_metrics.get("threshold"),
        "calibrated_threshold": summary.get("threshold_calibration", {}).get("threshold"),
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy": test_metrics["balanced_accuracy"],
        "precision_lie": test_metrics["precision_lie"],
        "recall_lie": test_metrics["recall_lie"],
        "f1_lie": test_metrics["f1_lie"],
        "macro_f1": test_metrics["macro_f1"],
        "auc_roc": test_metrics["auc_roc"],
        "auc_pr": test_metrics["auc_pr"],
        "checkpoint": summary["checkpoint"],
        "metrics_json": str(Path(config["outputs"]["metrics_dir"]) / f"{summary['fold']}_seed{summary['seed']}_test_metrics.json"),
        "config_path": args.config,
        "notes": args.journal_notes,
    }
    append_experiment_rows(csv_path, [row])
    render_experiment_journal(csv_path, md_path)
    return csv_path, md_path


def train(
    config: dict[str, Any],
    fold: str,
    seed: int,
    device_name: str | None,
    disable_wandb: bool,
    logger,
) -> dict[str, Any]:
    set_seed(seed)
    config["experiment"]["seed"] = seed
    train_csv = split_csv_from_config(config, "train", fold)
    val_csv = split_csv_from_config(config, "val", fold)
    test_csv = split_csv_from_config(config, "test", fold)
    train_loader = make_loader(config, train_csv, "dolos", train=True, seed=seed)
    val_loader = make_loader(config, val_csv, "dolos", train=False, seed=seed)
    test_loader = make_loader(config, test_csv, "dolos", train=False, seed=seed)

    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    criterion, loss_name, imbalance = choose_loss(config, train_csv)
    criterion = criterion.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["training"]["lr_head"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    amp = bool(config["training"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    out_dir = ensure_dir(config["outputs"]["checkpoint_dir"])
    metrics_dir = ensure_dir(config["outputs"]["metrics_dir"])
    logger.info("Training config=%s fold=%s seed=%s device=%s", config.get("experiment", {}).get("name"), fold, seed, device)
    logger.info("Train CSV=%s | Val CSV=%s | Test CSV=%s", train_csv, val_csv, test_csv)
    logger.info(
        "Loader sizes: train=%d val=%d test=%d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    logger.info("Loss=%s imbalance=%.4f", loss_name, imbalance)
    selection_metric_config = str(config["training"].get("early_stopping_metric", "val_f1_lie"))
    selection_metric_key = normalize_metric_name(selection_metric_config)
    threshold_metric = str(config["training"].get("threshold_calibration_metric", "balanced_accuracy"))
    log_feature_norms = bool(config["training"].get("log_feature_norms", False))
    feature_norm_batches = int(config["training"].get("feature_norm_batches", 1))
    aux_loss_weight = float(config["training"].get("aux_loss_weight", 0.0))
    aux_loss_values = config["training"].get("aux_loss_weights")
    aux_loss_weights = (
        tuple(float(value) for value in aux_loss_values)
        if aux_loss_values
        else (aux_loss_weight, aux_loss_weight, aux_loss_weight)
    )
    if len(aux_loss_weights) != 3:
        raise ValueError("training.aux_loss_weights must contain three values: spatial, flow, audio.")
    gate_entropy_weight = float(config["training"].get("gate_entropy_weight", 0.0))
    gate_prior_weight = float(config["training"].get("gate_prior_weight", 0.0))
    gate_prior_mode = str(config["training"].get("gate_prior_mode", "fixed"))
    gate_prior_weights = tuple(float(value) for value in config["training"].get("gate_prior_weights", []))
    gate_prior_low_face_weights = tuple(float(value) for value in config["training"].get("gate_prior_low_face_weights", []))
    gate_prior_high_face_weights = tuple(float(value) for value in config["training"].get("gate_prior_high_face_weights", []))
    gate_prior_face_power = float(config["training"].get("gate_prior_face_power", 1.0))
    gate_prior_sample_weight = str(config["training"].get("gate_prior_sample_weight", "face_valid"))
    face_valid_mode = str(config["data"].get("face_valid_mode", "binary"))
    logger.info("Checkpoint selection metric=%s", selection_metric_key)
    logger.info("Threshold calibration metric=%s", threshold_metric)
    logger.info("Face-valid mode=%s", face_valid_mode)
    logger.info("Feature norm logging=%s batches=%d", log_feature_norms, feature_norm_batches)
    logger.info("Aux loss weight=%.4f aux loss weights=%s gate entropy weight=%.4f", aux_loss_weight, aux_loss_weights, gate_entropy_weight)
    logger.info(
        "Gate prior mode=%s weight=%.4f weights=%s low_face=%s high_face=%s face_power=%.4f sample_weight=%s",
        gate_prior_mode,
        gate_prior_weight,
        gate_prior_weights,
        gate_prior_low_face_weights,
        gate_prior_high_face_weights,
        gate_prior_face_power,
        gate_prior_sample_weight,
    )
    log_path = ensure_dir(config["outputs"]["log_dir"]) / f"{fold}_seed{seed}_{selection_metric_key}_train_log.csv"
    best_path = out_dir / f"{fold}_seed{seed}_best.pt"
    run = maybe_init_wandb(config, disable_wandb, logger)

    best_metric = -1.0
    best_epoch = 0
    stale = 0
    patience = int(config["training"]["early_stopping_patience"])
    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            int(config["training"]["gradient_accumulation"]),
            amp,
            aux_loss_weight,
            gate_entropy_weight,
            gate_prior_weight,
            gate_prior_weights,
            aux_loss_weights,
            face_valid_mode,
            gate_prior_mode,
            gate_prior_low_face_weights,
            gate_prior_high_face_weights,
            gate_prior_face_power,
            gate_prior_sample_weight,
        )
        scheduler.step(epoch)
        val_windows = predict_windows(model, val_loader, device, amp, face_valid_mode)
        val_metrics = mean_metrics(val_windows)
        _, current = metric_value(val_metrics, selection_metric_key)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_selected_metric": current,
            "val_f1_lie": val_metrics["f1_lie"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_precision_lie": val_metrics["precision_lie"],
            "val_recall_lie": val_metrics["recall_lie"],
            "val_auc_roc": val_metrics["auc_roc"],
            "val_threshold": val_metrics["threshold"],
            "lr": optimizer.param_groups[0]["lr"],
            **confusion_columns(val_metrics, "val"),
        }
        if log_feature_norms:
            row.update(collect_feature_norms(model, val_loader, device, amp, max_batches=feature_norm_batches, face_valid_mode=face_valid_mode))
        append_log(log_path, row)
        if run is not None:
            run.log(row)
        logger.info("Epoch metrics: %s", row)
        if current > best_metric:
            best_metric = current
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": config,
                    "fold": fold,
                    "seed": seed,
                    "val_metrics": val_metrics,
                    "selection_metric": selection_metric_key,
                    "selection_metric_value": current,
                },
                best_path,
            )
            val_windows.to_csv(metrics_dir / f"{fold}_seed{seed}_val_window_predictions.csv", index=False)
            aggregate_windows(val_windows)["mean"].to_csv(metrics_dir / f"{fold}_seed{seed}_val_predictions.csv", index=False)
            write_json(metrics_dir / f"{fold}_seed{seed}_val_metrics.json", val_metrics)
            logger.info("New best checkpoint saved: epoch=%d %s=%.6f path=%s", epoch, selection_metric_key, current, best_path)
        else:
            stale += 1
            if stale >= patience:
                logger.info("Early stopping triggered at epoch %d after %d stale epochs.", epoch, stale)
                break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_windows = predict_windows(model, val_loader, device, amp, face_valid_mode)
    val_mean = aggregate_windows(val_windows)["mean"]
    calibration = find_best_threshold(
        val_mean["label"].to_numpy(),
        val_mean["score_lie"].to_numpy(),
        metric=threshold_metric,
    )
    calibrated_threshold = float(calibration["threshold"])
    val_metrics_default = binary_metrics(val_mean["label"].to_numpy(), val_mean["score_lie"].to_numpy())
    val_metrics_calibrated = binary_metrics(
        val_mean["label"].to_numpy(),
        val_mean["score_lie"].to_numpy(),
        threshold=calibrated_threshold,
    )
    val_windows.to_csv(metrics_dir / f"{fold}_seed{seed}_val_window_predictions.csv", index=False)
    val_mean.to_csv(metrics_dir / f"{fold}_seed{seed}_val_predictions.csv", index=False)
    write_json(metrics_dir / f"{fold}_seed{seed}_val_metrics.json", val_metrics_default)
    write_json(metrics_dir / f"{fold}_seed{seed}_val_metrics_calibrated.json", val_metrics_calibrated)

    test_windows = predict_windows(model, test_loader, device, amp, face_valid_mode)
    test_mean = aggregate_windows(test_windows)["mean"]
    test_metrics_default = binary_metrics(test_mean["label"].to_numpy(), test_mean["score_lie"].to_numpy())
    test_metrics_calibrated = binary_metrics(
        test_mean["label"].to_numpy(),
        test_mean["score_lie"].to_numpy(),
        threshold=calibrated_threshold,
    )
    test_metrics = test_metrics_default
    test_metrics["bootstrap_95ci"] = bootstrap_ci(
        test_mean["label"].to_numpy(),
        test_mean["score_lie"].to_numpy(),
        seed=seed,
    )
    test_metrics["calibrated_threshold_metrics"] = test_metrics_calibrated
    test_metrics["threshold_calibration"] = calibration
    test_metrics["val_metrics_default_threshold"] = val_metrics_default
    test_metrics["val_metrics_calibrated_threshold"] = val_metrics_calibrated
    test_windows.to_csv(metrics_dir / f"{fold}_seed{seed}_test_window_predictions.csv", index=False)
    test_mean.to_csv(metrics_dir / f"{fold}_seed{seed}_test_predictions.csv", index=False)
    write_json(metrics_dir / f"{fold}_seed{seed}_test_metrics.json", test_metrics)

    best_val_metrics = checkpoint.get("val_metrics", val_metrics_default)
    summary = {
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_metric_name": selection_metric_key,
        "best_val_metric": best_metric,
        "best_val_f1_lie": best_val_metrics.get("f1_lie"),
        "best_val_auc_roc": best_val_metrics.get("auc_roc"),
        "best_val_balanced_accuracy": best_val_metrics.get("balanced_accuracy"),
        "checkpoint": str(best_path),
        "loss_name": loss_name,
        "label_imbalance": imbalance,
        "threshold_calibration": calibration,
        "test_metrics_calibrated_threshold": test_metrics_calibrated,
        "test_metrics": test_metrics,
    }
    write_json(metrics_dir / f"{fold}_seed{seed}_summary.json", summary)
    if run is not None:
        run.finish()
        logger.info("W&B run finished.")
    logger.info("Training summary: %s", summary)
    return summary


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    seed = int(args.seed if args.seed is not None else config["experiment"].get("seed", 42))
    run_log_dir = args.log_dir or config.get("outputs", {}).get("log_dir", "outputs/logs")
    logger, log_path = setup_logger("train_multimodal", run_log_dir, f"train_multimodal_{args.fold}_seed{seed}.log")
    logger.info("Log file: %s", log_path)
    log_args(logger, args)
    summary = train(config, args.fold, seed, args.device, args.no_wandb, logger)
    journal_csv, journal_md = append_train_journal(
        args.journal_root,
        config,
        args,
        summary,
        summary["loss_name"],
        summary["label_imbalance"],
    )
    logger.info("Experiment journal updated: %s", journal_csv)
    logger.info("Experiment journal markdown updated: %s", journal_md)
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
