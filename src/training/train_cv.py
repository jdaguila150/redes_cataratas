"""5-Fold Stratified Cross-Validation for MobileNetV3-Small and EfficientNet-B0."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from src.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    CHECKPOINTS_DIR,
    EPOCHS,
    LEARNING_RATE,
    N_FOLDS,
    SEED,
    TEST_DIR,
    TRAIN_DIR,
    WEIGHT_DECAY,
)
from src.data.dataset import (
    CataractDataset,
    SampleRecord,
    build_preprocessed_cache,
    compute_pos_weight,
    filter_leakage_against,
    scan_split,
)
from src.evaluation.metrics import compute_triage_metrics
from src.models.cnn import SUPPORTED_BACKBONES, CataractClassifier, count_parameters
from src.preprocessing.pipeline import augment_normalized_rgb, preprocess_image


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x)).cpu().numpy()
            probs_all.append(probs)
            labels_all.append(y.numpy())
    return np.concatenate(labels_all), np.concatenate(probs_all)


def train_one_fold(
    backbone_name: str,
    fold_idx: int,
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    cache: dict[Path, np.ndarray],
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[dict[str, float], Path]:
    """Train a single CV fold and return best validation metrics + checkpoint path."""
    rng = np.random.default_rng(SEED + fold_idx)

    train_ds = CataractDataset(
        train_records,
        preprocess_fn=preprocess_image,
        augment_fn=lambda img: augment_normalized_rgb(img, rng=rng),
        cache=cache,
    )
    val_ds = CataractDataset(val_records, preprocess_fn=preprocess_image, augment_fn=None, cache=cache)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CataractClassifier(backbone_name=backbone_name, pretrained=True).to(device)

    train_labels = [r.label for r in train_records]
    pos_weight = torch.tensor([compute_pos_weight(train_labels)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_score = -1.0
    best_metrics: dict[str, float] = {}
    ckpt_path = checkpoint_dir / f"{backbone_name}_fold{fold_idx}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * x.size(0)
        scheduler.step()

        y_val, p_val = _predict_loader(model, val_loader, device)
        val_metrics = compute_triage_metrics(y_val, p_val, threshold=0.5)
        # Prioritize F2 and AUC-ROC for triage selection
        composite = 0.5 * val_metrics["f2"] + 0.5 * val_metrics["auc_roc"]

        if composite > best_score:
            best_score = composite
            best_metrics = val_metrics
            torch.save(
                {
                    "backbone": backbone_name,
                    "fold": fold_idx,
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )

        print(
            f"  [{backbone_name}][Fold {fold_idx}][Epoch {epoch:02d}/{epochs}] "
            f"loss={running_loss / len(train_ds):.4f} "
            f"sens={val_metrics['sensitivity']:.3f} spec={val_metrics['specificity']:.3f} "
            f"f2={val_metrics['f2']:.3f} auc={val_metrics['auc_roc']:.3f}"
        )

    return best_metrics, ckpt_path


def evaluate_ensemble_on_test(
    backbone_name: str,
    ckpt_paths: list[Path],
    test_records: list[SampleRecord],
    cache: dict[Path, np.ndarray],
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Average predicted probabilities across all 5 fold checkpoints on held-out test set."""
    test_ds = CataractDataset(test_records, preprocess_fn=preprocess_image, augment_fn=None, cache=cache)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    fold_probs: list[np.ndarray] = []
    y_true = np.array([r.label for r in test_records], dtype=np.int64)

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = CataractClassifier(backbone_name=backbone_name, pretrained=False).to(device)
        model.load_state_dict(ckpt["state_dict"])
        _, probs = _predict_loader(model, test_loader, device)
        fold_probs.append(probs)

    mean_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
    return compute_triage_metrics(y_true, mean_probs, threshold=0.5)


def run_cv_experiment(
    backbones: tuple[str, ...] = SUPPORTED_BACKBONES,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> dict:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    train_records, train_summary = scan_split(TRAIN_DIR, deduplicate=True)
    test_records_raw, test_summary = scan_split(TEST_DIR, deduplicate=True)
    test_records, leaked_removed = filter_leakage_against(test_records_raw, train_records)
    test_summary["leakage_removed_against_train"] = leaked_removed
    test_summary["final_test_count"] = len(test_records)

    print("Train summary:", train_summary)
    print("Test summary:", test_summary)
    print("Precomputing deterministic preprocessed cache...", flush=True)
    cache = build_preprocessed_cache(train_records + test_records, preprocess_image)
    print(f"Cached {len(cache)} preprocessed images.", flush=True)

    y_all = np.array([r.label for r in train_records])
    groups_all = np.array([r.group_hash for r in train_records])
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(np.zeros(len(train_records)), y_all, groups_all))

    results: dict = {
        "device": str(device),
        "train_summary": train_summary,
        "test_summary": test_summary,
        "models": {},
    }

    metric_keys = ("sensitivity", "specificity", "precision", "f2", "auc_roc")

    for backbone in backbones:
        t0 = time.time()
        tmp_model = CataractClassifier(backbone_name=backbone, pretrained=False)
        n_params = count_parameters(tmp_model)
        del tmp_model

        print(f"\n=== Training {backbone} (params={n_params:,}) ===")
        fold_metrics_list: list[dict[str, float]] = []
        ckpt_paths: list[Path] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(splits, start=1):
            tr_recs = [train_records[i] for i in tr_idx]
            val_recs = [train_records[i] for i in val_idx]
            fold_metrics, ckpt_path = train_one_fold(
                backbone_name=backbone,
                fold_idx=fold_idx,
                train_records=tr_recs,
                val_records=val_recs,
                cache=cache,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
                checkpoint_dir=CHECKPOINTS_DIR,
            )
            fold_metrics_list.append(fold_metrics)
            ckpt_paths.append(ckpt_path)

        cv_summary = {
            k: {
                "mean": float(np.mean([m[k] for m in fold_metrics_list])),
                "std": float(np.std([m[k] for m in fold_metrics_list])),
            }
            for k in metric_keys
        }
        test_metrics = evaluate_ensemble_on_test(
            backbone_name=backbone,
            ckpt_paths=ckpt_paths,
            test_records=test_records,
            cache=cache,
            batch_size=batch_size,
            device=device,
        )
        elapsed = time.time() - t0

        results["models"][backbone] = {
            "params": n_params,
            "elapsed_seconds": round(elapsed, 2),
            "folds": fold_metrics_list,
            "cv_summary": cv_summary,
            "test_ensemble_metrics": test_metrics,
        }
        print(f"=== {backbone} CV Summary ===")
        for k in metric_keys:
            print(f"  {k}: {cv_summary[k]['mean']:.4f} +/- {cv_summary[k]['std']:.4f}")
        print(f"=== {backbone} Held-out Test Ensemble ===")
        for k in metric_keys:
            print(f"  {k}: {test_metrics[k]:.4f}")

    out_path = ARTIFACTS_DIR / "results_cv.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved full results to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Fold CV for Cataract Triage CNNs")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=list(SUPPORTED_BACKBONES),
        choices=list(SUPPORTED_BACKBONES),
    )
    args = parser.parse_args()
    run_cv_experiment(
        backbones=tuple(args.backbones),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
