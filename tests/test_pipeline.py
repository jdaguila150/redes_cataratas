"""Unit tests for preprocessing, dataset labeling, lightweight CNNs, and triage metrics."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.config import ARTIFACTS_DIR, IMAGE_SIZE
from src.data.dataset import compute_pos_weight
from src.evaluation.metrics import compute_triage_metrics
from src.models.cnn import CataractClassifier
from src.preprocessing.pipeline import (
    apply_clahe,
    crop_pupillary_roi,
    preprocess_image,
    remove_glare,
)
from src.xai.gradcam import GradCAM


def test_preprocessing_pipeline_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    synthetic_bgr = rng.integers(0, 255, size=(320, 480, 3), dtype=np.uint8)
    # Add a bright specular spot to exercise glare removal
    synthetic_bgr[100:106, 120:126] = 255

    roi = crop_pupillary_roi(synthetic_bgr)
    assert roi.ndim == 3 and roi.shape[2] == 3

    deglared = remove_glare(roi)
    assert deglared.shape == roi.shape

    enhanced = apply_clahe(deglared)
    assert enhanced.shape == roi.shape

    out = preprocess_image(synthetic_bgr)
    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_cnn_backbones_output_shape_and_probabilities() -> None:
    x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    for backbone in ("mobilenet_v3_small", "efficientnet_b0"):
        model = CataractClassifier(backbone_name=backbone, pretrained=False)
        model.eval()
        logits = model(x)
        probs = model.predict_proba(x)
        assert logits.shape == (2,)
        assert probs.shape == (2,)
        assert torch.all((probs >= 0.0) & (probs <= 1.0))


def test_triage_metrics_and_pos_weight() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.4, 0.8, 0.9]
    metrics = compute_triage_metrics(y_true, y_prob, threshold=0.5)
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f2"] == 1.0
    assert metrics["auc_roc"] == 1.0

    assert compute_pos_weight([0, 0, 0, 1]) == 3.0


def test_gradcam_and_exported_onnx() -> None:
    model = CataractClassifier(backbone_name="mobilenet_v3_small", pretrained=False)
    cam = GradCAM(model)
    try:
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        prob, heatmap = cam.generate(x)
        assert 0.0 <= prob <= 1.0
        assert heatmap.shape == (IMAGE_SIZE, IMAGE_SIZE)
        assert np.isfinite(heatmap).all()
    finally:
        cam.close()

    onnx_path = ARTIFACTS_DIR / "export" / "cataract_mobilenet_v3_small_fp32.onnx"
    assert onnx_path.exists()
    assert onnx_path.stat().st_size < 5 * 1024 * 1024

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    logit, conv_feat = sess.run(None, {sess.get_inputs()[0].name: dummy})
    assert logit.shape == (1,)
    assert conv_feat.shape == (1, 576, 7, 7)
    assert np.isfinite(logit).all() and np.isfinite(conv_feat).all()


def test_web_app_predict_endpoint() -> None:
    import base64
    import cv2
    from fastapi.testclient import TestClient
    from src.web.app import app

    client = TestClient(app)
    resp_home = client.get("/")
    assert resp_home.status_code == 200

    synthetic_bgr = np.full((128, 128, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", synthetic_bgr)
    assert ok
    b64 = "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

    resp = client.post("/predict", json={"image_base64": b64})
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in {"NORMAL", "CATARACT"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["preprocessed_b64"].startswith("data:image/png;base64,")
    assert body["gradcam_overlay_b64"].startswith("data:image/png;base64,")
