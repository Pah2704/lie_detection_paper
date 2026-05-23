from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.models.fusion_cross_attention import ThreeStreamModel
from src.utils import ensure_dir, log_args, read_yaml, setup_logger, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test multimodal model VRAM usage.")
    parser.add_argument("--config", default="configs/multimodal_dolos.yaml")
    parser.add_argument("--batches", default="1,2,4,8", help="Clip batch sizes to test.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="outputs/metrics/estimate_vram.json")
    parser.add_argument("--no-backward", action="store_true", help="Forward-only smoke test.")
    parser.add_argument("--log-dir", default="outputs/logs")
    return parser.parse_args()


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
    )


def allocated_mb() -> float:
    return float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0


def test_batch(model: torch.nn.Module, config: dict[str, Any], clip_batch: int, device: torch.device, backward: bool) -> dict[str, Any]:
    data_cfg = config["data"]
    windows_per_clip = int(data_cfg["num_windows_per_clip_train"])
    batch_windows = clip_batch * windows_per_clip
    frames = int(data_cfg["frames_per_clip"])
    image_size = int(data_cfg["image_size"])
    audio_samples = int(round(float(data_cfg["window_seconds"]) * int(data_cfg["sample_rate"])))

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    rgb = torch.randn(batch_windows, frames, 3, image_size, image_size, device=device)
    flow = torch.randn(batch_windows, frames, 2, image_size, image_size, device=device)
    audio = torch.randn(batch_windows, audio_samples, device=device)
    labels = torch.randint(0, 2, (batch_windows,), device=device)
    criterion = torch.nn.CrossEntropyLoss()

    try:
        logits = model(rgb, flow, audio)
        loss = criterion(logits, labels)
        if backward:
            loss.backward()
        status = "ok"
        error = ""
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "failed"
        error = str(exc)
    peak = allocated_mb()
    del rgb, flow, audio, labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "clip_batch": clip_batch,
        "windows_per_clip": windows_per_clip,
        "batch_windows": batch_windows,
        "backward": backward,
        "status": status,
        "peak_cuda_mb": peak,
        "error": error,
    }


def main() -> None:
    args = parse_args()
    logger, log_path = setup_logger("estimate_vram", args.log_dir)
    logger.info("Log file: %s", log_path)
    log_args(logger, args)
    config = read_yaml(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config).to(device)
    model.train()
    backward = not args.no_backward
    results = []
    for batch in [int(x.strip()) for x in args.batches.split(",") if x.strip()]:
        result = test_batch(model, config, batch, device, backward)
        results.append(result)
        logger.info("Batch result: %s", result)
        if result["status"] != "ok":
            break
    out = Path(args.out)
    ensure_dir(out.parent)
    write_json(out, {"device": str(device), "results": results})
    logger.info("Wrote VRAM estimate to %s", out)


if __name__ == "__main__":
    main()
