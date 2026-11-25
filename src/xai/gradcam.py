"""Grad-CAM explainability (XAI) for CataractClassifier."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.config import ARTIFACTS_DIR, CHECKPOINTS_DIR, IMAGE_SIZE, TEST_DIR
from src.data.dataset import filter_leakage_against, scan_split
from src.config import TRAIN_DIR
from src.models.cnn import CataractClassifier
from src.preprocessing.pipeline import preprocess_image


class GradCAM:
    """Computes Grad-CAM heatmaps from the last convolutional block of CataractClassifier."""

    def __init__(self, model: CataractClassifier) -> None:
        self.model = model
        self.model.eval()
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        target_layer = self.model.features[-1]
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out: torch.Tensor) -> None:
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out: tuple[torch.Tensor, ...]) -> None:
        self._gradients = grad_out[0].detach()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor) -> tuple[float, np.ndarray]:
        """Return (cataract_probability, normalized_heatmap_2d) for a (1, 3, H, W) tensor."""
        self.model.zero_grad(set_to_none=True)
        logit = self.model(input_tensor)
        prob = float(torch.sigmoid(logit).item())
        logit.backward()

        assert self._activations is not None and self._gradients is not None
        # Global average pool gradients over spatial dimensions -> channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam_resized = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )
        heatmap = cam_resized.squeeze().cpu().numpy()
        max_val = float(heatmap.max())
        if max_val > 1e-8:
            heatmap = heatmap / max_val
        return prob, heatmap.astype(np.float32)


def _denormalize_to_uint8(norm_rgb: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = np.clip((norm_rgb * std + mean) * 255.0, 0, 255).astype(np.uint8)
    return img


def render_gradcam_triptych(
    preprocessed_rgb: np.ndarray,
    heatmap: np.ndarray,
    true_label: int,
    prob: float,
    out_path: Path,
) -> None:
    """Save side-by-side Input | Grad-CAM | Overlay visualization."""
    base_rgb = _denormalize_to_uint8(preprocessed_rgb)
    heatmap_u8 = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    jet_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(base_rgb, 0.55, jet_rgb, 0.45, 0)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    titles = ["Preprocessed Input", "Grad-CAM Heatmap", "Overlay"]
    for ax, img, title in zip(axes, (base_rgb, jet_rgb, overlay), titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    gt_str = "CATARACT" if true_label == 1 else "NORMAL"
    pred_str = "CATARACT" if prob >= 0.5 else "NORMAL"
    fig.suptitle(f"GT: {gt_str} | Pred: {pred_str} (p={prob:.3f})", fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_test_gradcams(
    backbone_name: str = "mobilenet_v3_small",
    fold: int = 5,
    max_samples_per_class: int = 4,
) -> list[str]:
    """Generate Grad-CAM triptychs on clean held-out test images."""
    ckpt_path = CHECKPOINTS_DIR / f"{backbone_name}_fold{fold}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CataractClassifier(backbone_name=backbone_name, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])

    train_records, _ = scan_split(TRAIN_DIR, deduplicate=True)
    test_records_raw, _ = scan_split(TEST_DIR, deduplicate=True)
    test_records, _ = filter_leakage_against(test_records_raw, train_records)

    out_dir = ARTIFACTS_DIR / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = GradCAM(model)
    saved_files: list[str] = []
    counts = {0: 0, 1: 0}

    try:
        for rec in test_records:
            if counts[rec.label] >= max_samples_per_class:
                continue
            img_bgr = cv2.imread(str(rec.path))
            if img_bgr is None:
                continue
            proc_rgb = preprocess_image(img_bgr)
            tensor = torch.from_numpy(np.ascontiguousarray(proc_rgb.transpose(2, 0, 1))).unsqueeze(0).float()
            prob, heatmap = cam.generate(tensor)

            cls_name = "cataract" if rec.label == 1 else "normal"
            out_path = out_dir / f"{cls_name}_{counts[rec.label] + 1}_{rec.path.stem}.png"
            render_gradcam_triptych(proc_rgb, heatmap, rec.label, prob, out_path)
            saved_files.append(str(out_path))
            counts[rec.label] += 1

            if all(c >= max_samples_per_class for c in counts.values()):
                break
    finally:
        cam.close()

    manifest = {"backbone": backbone_name, "fold": fold, "generated_files": saved_files}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated {len(saved_files)} Grad-CAM visualizations in {out_dir}")
    return saved_files


if __name__ == "__main__":
    generate_test_gradcams()
