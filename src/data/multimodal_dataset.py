from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass(frozen=True)
class WindowRecord:
    clip_idx: int
    start_time: float | None
    window_index: int
    num_windows: int


class MultimodalClipDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        dataset_name: str,
        audio_root: str | Path = "data/processed/audio",
        faces_root: str | Path = "data/processed/faces_224_clean",
        face_valid_root: str | Path | None = None,
        optflow_root: str | Path = "data/processed/optflow",
        train: bool = False,
        frames_per_clip: int = 16,
        window_seconds: float = 2.0,
        num_windows_per_clip_train: int = 3,
        sliding_stride_seconds: float = 1.0,
        max_windows_per_clip: int = 16,
        min_window_start_seconds: float = 0.0,
        min_window_face_valid_ratio: float = 0.75,
        use_face_valid_mask: bool = True,
        sample_rate: int = 16000,
        target_fps: float = 25.0,
        image_size: int = 224,
        speed_perturb_range: tuple[float, float] = (0.9, 1.1),
        random_erasing_prob: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.df = pd.read_csv(split_csv).reset_index(drop=True)
        if "video_id" not in self.df.columns:
            raise ValueError("split_csv must contain video_id")
        if "label" not in self.df.columns:
            raise ValueError("split_csv must contain label")
        self.dataset_name = dataset_name
        self.audio_root = Path(audio_root)
        self.faces_root = Path(faces_root)
        self.face_valid_root = Path(face_valid_root) if face_valid_root else self.faces_root / self.dataset_name / "face_valid"
        self.optflow_root = Path(optflow_root)
        self.train = train
        self.frames_per_clip = frames_per_clip
        self.window_seconds = float(window_seconds)
        self.num_windows_per_clip_train = int(num_windows_per_clip_train)
        self.sliding_stride_seconds = float(sliding_stride_seconds)
        self.max_windows_per_clip = int(max_windows_per_clip)
        self.min_window_start_seconds = max(0.0, float(min_window_start_seconds))
        self.min_window_face_valid_ratio = float(min_window_face_valid_ratio)
        self.use_face_valid_mask = bool(use_face_valid_mask)
        self.sample_rate = int(sample_rate)
        self.target_fps = float(target_fps)
        self.image_size = int(image_size)
        self.speed_perturb_range = speed_perturb_range
        self.random_erasing_prob = float(random_erasing_prob)
        self.seed = int(seed)
        self.frame_paths = [self._frame_paths(str(row.video_id), row) for row in self.df.itertuples()]
        self.face_validity = [
            self._face_validity(str(row.video_id), len(frames))
            for row, frames in zip(self.df.itertuples(), self.frame_paths, strict=True)
        ]
        self.records = self._build_records()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        row = self.df.iloc[record.clip_idx]
        video_id = str(row["video_id"])
        label = int(row["label"])
        frames = self.frame_paths[record.clip_idx]
        if not frames:
            raise FileNotFoundError(f"No frames found for {video_id}")

        duration = len(frames) / self.target_fps
        rng = np.random.default_rng(self.seed + idx + (0 if not self.train else np.random.randint(0, 2**16)))
        if self.train:
            starts = [self._random_train_start(duration, rng) for _ in range(record.num_windows)]
        else:
            assert record.start_time is not None
            starts = [record.start_time]

        rgb_windows: list[torch.Tensor] = []
        flow_windows: list[torch.Tensor] = []
        audio_windows: list[torch.Tensor] = []
        face_valid_flags: list[float] = []
        face_valid_ratios: list[float] = []
        face_valid_frames: list[torch.Tensor] = []
        validity = self.face_validity[record.clip_idx]
        for start_time in starts:
            frame_indices = self._sample_frame_indices(len(frames), start_time, duration)
            rgb_windows.append(self._load_rgb_frames(frames, frame_indices, rng))
            flow_windows.append(self._load_flow(video_id, frame_indices))
            audio_windows.append(self._load_audio(video_id, start_time, rng))
            valid_flag, valid_ratio, valid_frames = self._window_face_valid(validity, frame_indices)
            face_valid_flags.append(valid_flag)
            face_valid_ratios.append(valid_ratio)
            face_valid_frames.append(torch.from_numpy(valid_frames))

        return {
            "rgb_frames": torch.stack(rgb_windows, dim=0),
            "optflow_frames": torch.stack(flow_windows, dim=0),
            "audio_waveform": torch.stack(audio_windows, dim=0),
            "label": torch.tensor(label, dtype=torch.long),
            "video_id": video_id,
            "window_index": torch.arange(len(starts), dtype=torch.long) if self.train else torch.tensor([record.window_index], dtype=torch.long),
            "num_windows": torch.tensor(record.num_windows, dtype=torch.long),
            "start_time": torch.tensor(starts, dtype=torch.float32),
            "face_valid": torch.tensor(face_valid_flags, dtype=torch.float32),
            "face_valid_ratio": torch.tensor(face_valid_ratios, dtype=torch.float32),
            "face_valid_frames": torch.stack(face_valid_frames, dim=0).float(),
        }

    def _frame_paths(self, video_id: str, row: Any) -> list[Path]:
        if hasattr(row, "frames_dir") and isinstance(row.frames_dir, str):
            frames_dir = Path(row.frames_dir)
        else:
            frames_dir = self.faces_root / self.dataset_name / video_id
        return sorted(frames_dir.glob("*.jpg"))

    def _face_validity(self, video_id: str, num_frames: int) -> np.ndarray:
        if not self.use_face_valid_mask or num_frames <= 0:
            return np.ones(max(num_frames, 1), dtype=np.float32)
        path = self.face_valid_root / f"{video_id}.csv"
        if not path.exists():
            return np.ones(num_frames, dtype=np.float32)
        try:
            df = pd.read_csv(path)
        except Exception:
            return np.ones(num_frames, dtype=np.float32)
        if "face_valid" in df.columns:
            values = df["face_valid"].astype(float).to_numpy(dtype=np.float32)
        elif "valid" in df.columns:
            values = df["valid"].astype(float).to_numpy(dtype=np.float32)
        else:
            return np.ones(num_frames, dtype=np.float32)
        validity = np.zeros(num_frames, dtype=np.float32)
        if "frame_index" in df.columns:
            frame_indices = df["frame_index"].astype(int).to_numpy()
            keep = (frame_indices >= 0) & (frame_indices < num_frames)
            validity[frame_indices[keep]] = values[keep]
        else:
            count = min(num_frames, len(values))
            validity[:count] = values[:count]
        return np.clip(validity, 0.0, 1.0)

    def _build_records(self) -> list[WindowRecord]:
        records: list[WindowRecord] = []
        for clip_idx, frames in enumerate(self.frame_paths):
            duration = len(frames) / self.target_fps if frames else 0.0
            if self.train:
                repeats = 1 if duration <= self.window_seconds else self.num_windows_per_clip_train
                records.append(WindowRecord(clip_idx, None, 0, repeats))
            else:
                starts = self._eval_starts(duration)
                for window_idx, start in enumerate(starts):
                    records.append(WindowRecord(clip_idx, float(start), window_idx, len(starts)))
        return records

    def _eval_starts(self, duration: float) -> np.ndarray:
        if duration <= self.window_seconds:
            return np.array([0.0], dtype=np.float32)
        last_start = max(duration - self.window_seconds, 0.0)
        starts = np.arange(0.0, last_start + 1e-6, self.sliding_stride_seconds, dtype=np.float32)
        if starts.size == 0 or abs(float(starts[-1]) - last_start) > 1e-3:
            starts = np.concatenate([starts, np.array([last_start], dtype=np.float32)])
        starts = np.unique(np.round(starts, 4))
        if self.min_window_start_seconds > 0.0 and last_start >= self.min_window_start_seconds:
            starts = starts[starts >= self.min_window_start_seconds]
            if starts.size == 0:
                starts = np.array([last_start], dtype=np.float32)
        if len(starts) > self.max_windows_per_clip:
            keep = np.linspace(0, len(starts) - 1, self.max_windows_per_clip).round().astype(int)
            starts = starts[keep]
        return starts.astype(np.float32)

    def _random_train_start(self, duration: float, rng: np.random.Generator) -> float:
        if duration <= self.window_seconds:
            return 0.0
        last_start = duration - self.window_seconds
        first_start = self.min_window_start_seconds if last_start >= self.min_window_start_seconds else 0.0
        return float(rng.uniform(first_start, last_start))

    def _sample_frame_indices(self, num_frames: int, start_time: float, duration: float) -> np.ndarray:
        if num_frames <= 1:
            return np.zeros(self.frames_per_clip, dtype=np.int64)
        if duration <= self.window_seconds:
            start_frame = 0
            end_frame = num_frames - 1
        else:
            start_frame = int(round(start_time * self.target_fps))
            end_frame = int(round((start_time + self.window_seconds) * self.target_fps)) - 1
            start_frame = int(np.clip(start_frame, 0, num_frames - 1))
            end_frame = int(np.clip(end_frame, start_frame, num_frames - 1))
        return np.linspace(start_frame, end_frame, self.frames_per_clip).round().astype(np.int64)

    def _window_face_valid(self, validity: np.ndarray, indices: np.ndarray) -> tuple[float, float, np.ndarray]:
        if validity.size == 0:
            return 1.0, 1.0, np.ones(len(indices), dtype=np.float32)
        selected = validity[np.clip(indices, 0, len(validity) - 1)]
        valid_frames = (selected >= 0.5).astype(np.float32)
        valid_ratio = float(np.mean(valid_frames))
        valid_flag = 1.0 if valid_ratio >= self.min_window_face_valid_ratio else 0.0
        return valid_flag, valid_ratio, valid_frames

    def _load_rgb_frames(self, frame_paths: list[Path], indices: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
        tensors: list[torch.Tensor] = []
        last_image: np.ndarray | None = None
        for idx in indices:
            image = cv2.imread(str(frame_paths[int(np.clip(idx, 0, len(frame_paths) - 1))]), cv2.IMREAD_COLOR)
            if image is None:
                image = last_image if last_image is not None else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            last_image = image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[:2] != (self.image_size, self.image_size):
                image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
            tensors.append(tensor)
        rgb = torch.stack(tensors, dim=0)
        if self.train and rng.random() < self.random_erasing_prob:
            rgb = self._random_erase(rgb, rng)
        return rgb

    def _random_erase(self, rgb: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
        _, _, height, width = rgb.shape
        erase_h = int(rng.integers(max(1, height // 12), max(2, height // 4)))
        erase_w = int(rng.integers(max(1, width // 12), max(2, width // 4)))
        top = int(rng.integers(0, max(1, height - erase_h + 1)))
        left = int(rng.integers(0, max(1, width - erase_w + 1)))
        rgb = rgb.clone()
        rgb[:, :, top : top + erase_h, left : left + erase_w] = 0.0
        return rgb

    def _load_flow(self, video_id: str, indices: np.ndarray) -> torch.Tensor:
        path = self.optflow_root / self.dataset_name / f"{video_id}.npz"
        if not path.exists():
            return torch.zeros((self.frames_per_clip, 2, self.image_size, self.image_size), dtype=torch.float32)
        with np.load(path) as data:
            flow = data["flow"]
        if flow.ndim != 4:
            raise ValueError(f"Expected flow shape (T,2,H,W), got {flow.shape} for {path}")
        clipped = np.clip(indices, 0, flow.shape[0] - 1)
        selected = torch.from_numpy(flow[clipped].astype(np.float32))
        if selected.shape[-2:] != (self.image_size, self.image_size):
            selected = F.interpolate(selected, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return selected

    def _load_audio(self, video_id: str, start_time: float, rng: np.random.Generator) -> torch.Tensor:
        expected = int(round(self.window_seconds * self.sample_rate))
        path = self.audio_root / self.dataset_name / f"{video_id}.wav"
        if not path.exists():
            return torch.zeros(expected, dtype=torch.float32)
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(path))
            waveform = waveform.mean(dim=0)
            if int(sr) != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, int(sr), self.sample_rate)
        except Exception:
            return torch.zeros(expected, dtype=torch.float32)

        start = int(round(start_time * self.sample_rate))
        segment = waveform[start : start + expected]
        if segment.numel() < expected:
            segment = F.pad(segment, (0, expected - segment.numel()))
        if self.train:
            low, high = self.speed_perturb_range
            rate = float(rng.uniform(low, high))
            if abs(rate - 1.0) > 1e-3 and segment.numel() > 1:
                new_len = max(1, int(round(segment.numel() / rate)))
                segment = F.interpolate(segment.view(1, 1, -1), size=new_len, mode="linear", align_corners=False).view(-1)
                if segment.numel() < expected:
                    segment = F.pad(segment, (0, expected - segment.numel()))
                segment = segment[:expected]
        return segment.float()


def multimodal_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    rgb = torch.cat([item["rgb_frames"] for item in batch], dim=0)
    flow = torch.cat([item["optflow_frames"] for item in batch], dim=0)
    audio = torch.cat([item["audio_waveform"] for item in batch], dim=0)
    labels: list[torch.Tensor] = []
    video_ids: list[str] = []
    window_indices: list[torch.Tensor] = []
    num_windows: list[torch.Tensor] = []
    start_times: list[torch.Tensor] = []
    face_valid: list[torch.Tensor] = []
    face_valid_ratios: list[torch.Tensor] = []
    face_valid_frames: list[torch.Tensor] = []
    for item in batch:
        count = item["rgb_frames"].shape[0]
        labels.append(item["label"].repeat(count))
        video_ids.extend([item["video_id"]] * count)
        window_indices.append(item["window_index"])
        num_windows.append(item["num_windows"].repeat(count))
        start_times.append(item["start_time"])
        face_valid.append(item["face_valid"])
        face_valid_ratios.append(item["face_valid_ratio"])
        face_valid_frames.append(item["face_valid_frames"])
    return {
        "rgb_frames": rgb,
        "optflow_frames": flow,
        "audio_waveform": audio,
        "label": torch.cat(labels, dim=0),
        "video_id": video_ids,
        "window_index": torch.cat(window_indices, dim=0),
        "num_windows": torch.cat(num_windows, dim=0),
        "start_time": torch.cat(start_times, dim=0),
        "face_valid": torch.cat(face_valid, dim=0),
        "face_valid_ratio": torch.cat(face_valid_ratios, dim=0),
        "face_valid_frames": torch.cat(face_valid_frames, dim=0),
    }
