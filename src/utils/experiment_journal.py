from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir


def append_experiment_rows(csv_path: str | Path, rows: list[dict[str, Any]]) -> Path:
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    new_df = pd.DataFrame(rows)
    if csv_path.exists():
        old_df = pd.read_csv(csv_path)
        columns = list(dict.fromkeys([*old_df.columns.tolist(), *new_df.columns.tolist()]))
        old_df = old_df.reindex(columns=columns)
        new_df = new_df.reindex(columns=columns)
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df
    out_df.to_csv(csv_path, index=False)
    return csv_path


def render_experiment_journal(csv_path: str | Path, md_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    md_path = Path(md_path)
    ensure_dir(md_path.parent)
    if not csv_path.exists():
        md_path.write_text("# Experiment Journal\n\nNo entries yet.\n", encoding="utf-8")
        return md_path

    df = pd.read_csv(csv_path).fillna("")
    preferred_columns = [
        "recorded_at",
        "stage",
        "experiment_name",
        "tag",
        "dataset",
        "fold",
        "seed",
        "aggregation",
        "stream_mode",
        "accuracy",
        "balanced_accuracy",
        "f1_lie",
        "macro_f1",
        "auc_roc",
        "best_val_metric_name",
        "best_val_metric",
        "best_val_f1_lie",
        "threshold",
        "calibrated_threshold",
        "notes",
    ]
    columns = [col for col in preferred_columns if col in df.columns]
    if not columns:
        columns = df.columns.tolist()
    view = df[columns].copy()
    view = view.sort_values(by=[col for col in ["recorded_at", "stage"] if col in view.columns], ascending=[False, True])
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(_escape_markdown_cell(row[col]) for col in columns) + " |"
        for _, row in view.iterrows()
    ]

    lines = [
        "# Experiment Journal",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        header,
        divider,
        *body,
        "",
        f"CSV source: `{csv_path}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _escape_markdown_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
