"""Export MobileNetV3-Small to ONNX (FP32 -> INT8) and benchmark mobile targets (<5 MB, <250 ms)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.config import ARTIFACTS_DIR, CHECKPOINTS_DIR, IMAGE_SIZE, TEST_DIR, TRAIN_DIR
from src.data.dataset import filter_leakage_against, scan_split
from src.evaluation.metrics import compute_triage_metrics
from src.models.cnn import CataractClassifier
from src.preprocessing.pipeline import preprocess_image


class ExportableCataractNet(nn.Module):
    """Wraps CataractClassifier to output both raw logit and last conv feature map for mobile Grad-CAM."""

    def __init__(self, base_model: CataractClassifier) -> None:
        super().__init__()
        self.features = base_model.features
        self.gap = base_model.gap
        self.linear = base_model.head[2]  # nn.Linear(576, 1) after Flatten & Dropout

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat_map = self.features(x)  # (B, 576, 7, 7)
        pooled = self.gap(feat_map).flatten(1)  # (B, 576)
        logit = self.linear(pooled).squeeze(-1)  # (B,)
        return logit, feat_map


class _CalibrationReader(CalibrationDataReader):
    """Feeds preprocessed training samples to calibrate INT8 activation ranges."""

    def __init__(self, input_name: str, samples: list[np.ndarray]) -> None:
        self._iter = iter({input_name: s} for s in samples)

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)


def _load_preprocessed_inputs(records: list) -> tuple[list[np.ndarray], list[int]]:
    inputs: list[np.ndarray] = []
    labels: list[int] = []
    for rec in records:
        img_bgr = cv2.imread(str(rec.path))
        if img_bgr is None:
            continue
        proc = preprocess_image(img_bgr)
        arr = np.ascontiguousarray(proc.transpose(2, 0, 1))[None, ...].astype(np.float32)
        inputs.append(arr)
        labels.append(rec.label)
    return inputs, labels
def _evaluate_onnx(session: ort.InferenceSession, test_inputs: list[np.ndarray], y_true: list[int]) -> dict:
    input_name = session.get_inputs()[0].name
    probs: list[float] = []
    latencies_ms: list[float] = []

    # Warmup
    session.run(None, {input_name: test_inputs[0]})

    for x in test_inputs:
        t0 = time.perf_counter()
        logit = session.run(None, {input_name: x})[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)
        prob = float(1.0 / (1.0 + np.exp(-float(logit[0]))))
        probs.append(prob)

    metrics = compute_triage_metrics(y_true, probs, threshold=0.5)
    metrics["mean_latency_ms"] = round(float(np.mean(latencies_ms)), 3)
    metrics["p95_latency_ms"] = round(float(np.percentile(latencies_ms, 95)), 3)
    return metrics


def export_and_benchmark(backbone_name: str = "mobilenet_v3_small", fold: int = 5) -> dict:
    ckpt_path = CHECKPOINTS_DIR / f"{backbone_name}_fold{fold}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_model = CataractClassifier(backbone_name=backbone_name, pretrained=False)
    base_model.load_state_dict(ckpt["state_dict"])
    base_model.eval()

    export_model = ExportableCataractNet(base_model).eval()

    export_dir = ARTIFACTS_DIR / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = export_dir / f"cataract_{backbone_name}_fp32.onnx"
    prep_path = export_dir / f"cataract_{backbone_name}_prep.onnx"
    int8_path = export_dir / f"cataract_{backbone_name}_int8.onnx"
    weights_path = export_dir / "classifier_weights.json"

    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        export_model,
        dummy,
        str(fp32_path),
        input_names=["input"],
        output_names=["logit", "conv_features"],
        opset_version=18,
        dynamo=False,
    )

    # Save linear classifier weights for zero-backprop Grad-CAM on Android
    linear_w = export_model.linear.weight.detach().cpu().numpy().flatten().tolist()
    linear_b = float(export_model.linear.bias.detach().cpu().item())
    weights_path.write_text(
        json.dumps({"weights": linear_w, "bias": linear_b, "num_channels": len(linear_w)}),
        encoding="utf-8",
    )

    # Load clean held-out test set and calibration subset from train
    train_records, _ = scan_split(TRAIN_DIR, deduplicate=True)
    test_records_raw, _ = scan_split(TEST_DIR, deduplicate=True)
    test_records, _ = filter_leakage_against(test_records_raw, train_records)

    calib_inputs, _ = _load_preprocessed_inputs(train_records[::6])  # ~62 balanced samples
    test_inputs, y_true = _load_preprocessed_inputs(test_records)

    quant_pre_process(str(fp32_path), str(prep_path))
    quantize_static(
        model_input=str(prep_path),
        model_output=str(int8_path),
        calibration_data_reader=_CalibrationReader("input", calib_inputs),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    fp32_mb = round(fp32_path.stat().st_size / (1024 * 1024), 3)
    int8_mb = round(int8_path.stat().st_size / (1024 * 1024), 3)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1  # Simulate conservative single-thread mobile CPU
    fp32_sess = ort.InferenceSession(str(fp32_path), sess_options=opts, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(str(int8_path), sess_options=opts, providers=["CPUExecutionProvider"])

    fp32_metrics = _evaluate_onnx(fp32_sess, test_inputs, y_true)
    int8_metrics = _evaluate_onnx(int8_sess, test_inputs, y_true)

    report = {
        "backbone": backbone_name,
        "fold": fold,
        "fp32_model": str(fp32_path),
        "int8_model": str(int8_path),
        "fp32_size_mb": fp32_mb,
        "int8_size_mb": int8_mb,
        "meets_size_target_lt_5mb": int8_mb < 5.0,
        "meets_latency_target_lt_250ms": int8_metrics["mean_latency_ms"] < 250.0,
        "fp32_test_metrics": fp32_metrics,
        "int8_test_metrics": int8_metrics,
    }
    report_path = export_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    export_and_benchmark()
