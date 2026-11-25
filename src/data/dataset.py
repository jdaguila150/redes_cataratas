"""Dataset discovery, integrity verification, deduplication, and PyTorch Dataset."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import POSITIVE_FOLDERS, VALID_EXTENSIONS


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int  # 0 = NORMAL, 1 = CATARACT
    group_hash: str


def _compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _compute_dhash(image_bgr: np.ndarray, hash_size: int = 8) -> str:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    bits = "".join("1" if v else "0" for v in diff.flatten())
    return hex(int(bits, 2))


def scan_split(split_dir: Path, deduplicate: bool = True) -> tuple[list[SampleRecord], dict]:
    """Scan a split directory (e.g., train/ or test/) and assign binary labels.

    Folders in POSITIVE_FOLDERS map to label 1 (CATARACT); all other folders map to 0 (NORMAL).
    Also checks image readability and optionally removes exact/perceptual duplicates within the split.
    """
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    records: list[SampleRecord] = []
    seen_sha: set[str] = set()
    corrupt_files: list[str] = []
    duplicate_files: list[str] = []

    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = 1 if class_dir.name.lower() in POSITIVE_FOLDERS else 0
        for img_path in sorted(class_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                corrupt_files.append(str(img_path))
                continue

            sha = _compute_file_sha256(img_path)
            if deduplicate and sha in seen_sha:
                duplicate_files.append(str(img_path))
                continue
            seen_sha.add(sha)

            dhash = _compute_dhash(img)
            records.append(SampleRecord(path=img_path, label=label, group_hash=dhash))

    summary = {
        "split_dir": str(split_dir),
        "total_valid": len(records),
        "normal_count": sum(1 for r in records if r.label == 0),
        "cataract_count": sum(1 for r in records if r.label == 1),
        "corrupt_removed": len(corrupt_files),
        "exact_duplicates_removed": len(duplicate_files),
    }
    return records, summary


def filter_leakage_against(
    target_records: Sequence[SampleRecord],
    reference_records: Sequence[SampleRecord],
) -> tuple[list[SampleRecord], int]:
    """Remove any target records whose perceptual hash matches a reference record."""
    ref_hashes = {r.group_hash for r in reference_records}
    filtered = [r for r in target_records if r.group_hash not in ref_hashes]
    removed = len(target_records) - len(filtered)
    return filtered, removed


def compute_pos_weight(labels: Sequence[int]) -> float:
    """Compute positive class weight (neg / pos) for BCEWithLogitsLoss."""
    pos = sum(1 for y in labels if y == 1)
    neg = sum(1 for y in labels if y == 0)
    if pos == 0:
        return 1.0
    return float(neg) / float(pos)


class CataractDataset(Dataset):
    """PyTorch Dataset applying deterministic preprocessing and optional augmentation."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        preprocess_fn: Callable[[np.ndarray], np.ndarray],
        augment_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        cache: dict[Path, np.ndarray] | None = None,
    ) -> None:
        self.records = list(records)
        self.preprocess_fn = preprocess_fn
        self.augment_fn = augment_fn
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        if self.cache is not None and rec.path in self.cache:
            processed_rgb = self.cache[rec.path]
        else:
            img_bgr = cv2.imread(str(rec.path))
            if img_bgr is None:
                raise RuntimeError(f"Failed to read image: {rec.path}")
            processed_rgb = self.preprocess_fn(img_bgr)
            if self.cache is not None:
                self.cache[rec.path] = processed_rgb

        if self.augment_fn is not None:
            processed_rgb = self.augment_fn(processed_rgb)

        # HWC -> CHW float32 tensor
        tensor = torch.from_numpy(np.ascontiguousarray(processed_rgb.transpose(2, 0, 1))).float()
        label = torch.tensor(rec.label, dtype=torch.float32)
        return tensor, label


def build_preprocessed_cache(
    records: Sequence[SampleRecord],
    preprocess_fn: Callable[[np.ndarray], np.ndarray],
) -> dict[Path, np.ndarray]:
    """Precompute deterministic preprocessed (224x224x3) arrays once in memory."""
    cache: dict[Path, np.ndarray] = {}
    for rec in records:
        if rec.path in cache:
            continue
        img_bgr = cv2.imread(str(rec.path))
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {rec.path}")
        cache[rec.path] = preprocess_fn(img_bgr)
    return cache
