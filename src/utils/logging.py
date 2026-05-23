from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir


def setup_logger(
    name: str,
    log_dir: str | Path = "outputs/logs",
    log_file: str | None = None,
    level: int = logging.INFO,
) -> tuple[logging.Logger, Path]:
    log_dir = ensure_dir(log_dir)
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{name}_{timestamp}.log"
    log_path = log_dir / log_file
    logger_name = f"lie_nonlie.{name}.{log_path.stem}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def log_args(logger: logging.Logger, args: Any) -> None:
    if hasattr(args, "__dict__"):
        logger.info("Arguments: %s", vars(args))
    else:
        logger.info("Arguments: %s", args)
