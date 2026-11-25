"""FastAPI web service for Cataract Triage (ONNX Runtime + Grad-CAM).

Run locally:
    uvicorn src.web.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import base64
import json
import time

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.config import ARTIFACTS_DIR, IMAGE_SIZE
from src.preprocessing.pipeline import preprocess_image

ONNX_PATH = ARTIFACTS_DIR / "export" / "cataract_mobilenet_v3_small_fp32.onnx"
WEIGHTS_PATH = ARTIFACTS_DIR / "export" / "classifier_weights.json"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(title="Triaje de Cataratas — ONNX + Grad-CAM")

_session: ort.InferenceSession | None = None
_classifier_weights: np.ndarray | None = None


def _get_engine() -> tuple[ort.InferenceSession, np.ndarray]:
    global _session, _classifier_weights
    if _session is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        _session = ort.InferenceSession(str(ONNX_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    if _classifier_weights is None:
        data = json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
        _classifier_weights = np.array(data["weights"], dtype=np.float32)
    return _session, _classifier_weights


def _encode_png_b64(rgb_u8: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _compute_gradcam_overlay(base_rgb_u8: np.ndarray, conv_feat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # conv_feat: (1, C, 7, 7)
    fmap = conv_feat[0]
    cam = np.tensordot(weights, fmap, axes=([0], [0]))  # (7, 7)
    cam = np.maximum(cam, 0.0)
    max_v = float(cam.max())
    if max_v > 1e-8:
        cam = cam / max_v
    cam_resized = cv2.resize(cam.astype(np.float32), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    heat_u8 = np.uint8(np.clip(cam_resized, 0.0, 1.0) * 255)
    jet_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base_rgb_u8, 0.55, jet_rgb, 0.45, 0)


class PredictRequest(BaseModel):
    image_base64: str


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Triaje de Cataratas (ONNX + Grad-CAM)</title>
  <style>
    body { font-family: system-ui, -apple-system, sans-serif; max-width: 780px; margin: 24px auto; padding: 0 16px; background: #f6f8fb; color: #1a2332; }
    .card { background: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,.06); margin-bottom: 16px; }
    h1 { font-size: 1.4rem; margin: 0 0 8px; }
    .sub { color: #516075; font-size: .92rem; margin: 0 0 16px; }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; }
    label.btn { background: #0f4c81; color: #fff; padding: 10px 16px; border-radius: 8px; cursor: pointer; font-weight: 600; display: inline-block; }
    input[type=file] { display: none; }
    .result { margin-top: 14px; padding: 14px; border-radius: 8px; font-weight: 600; display: none; }
    .pos { background: #fdecea; color: #9c1b1b; border: 1px solid #f5c2c0; }
    .neg { background: #eaf7ee; color: #166534; border: 1px solid #b7e4c7; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 16px; }
    .img-box { text-align: center; }
    .img-box img { width: 224px; height: 224px; object-fit: contain; border-radius: 8px; border: 1px solid #dce3ec; background: #eee; }
    .disc { font-size: .8rem; color: #64748b; text-align: center; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Sistema Inteligente para Triaje de Cataratas</h1>
    <p class="sub">MobileNetV3-Small (ONNX) + Explicabilidad Grad-CAM</p>
    <div class="controls">
      <label class="btn">Elegir imagen <input id="fileInput" type="file" accept="image/*" /></label>
      <label class="btn" style="background:#1b6ca8;">Usar cámara (móvil) <input id="camInput" type="file" accept="image/*" capture="environment" /></label>
    </div>
    <div id="status" class="result"></div>
    <div id="grid" class="grid" style="display:none;">
      <div class="img-box"><div><b>Preprocesada (ROI + CLAHE)</b></div><img id="imgPrep" /></div>
      <div class="img-box"><div><b>Grad-CAM (Overlay)</b></div><img id="imgCam" /></div>
    </div>
  </div>
  <p class="disc">Herramienta de apoyo al triaje médico. No sustituye la valoración de un oftalmólogo.</p>
  <script>
    async function handleFile(file) {
      if (!file) return;
      const status = document.getElementById('status');
      status.style.display = 'block';
      status.className = 'result';
      status.textContent = 'Analizando imagen...';
      const reader = new FileReader();
      reader.onload = async () => {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: reader.result })
        });
        const data = await res.json();
        if (!res.ok) { status.textContent = data.detail || 'Error'; return; }
        status.className = 'result ' + (data.prediction === 'CATARACT' ? 'pos' : 'neg');
        status.textContent = `${data.triage_message} | Probabilidad: ${(data.probability * 100).toFixed(1)}% | Latencia ONNX: ${data.latency_ms} ms`;
        document.getElementById('imgPrep').src = data.preprocessed_b64;
        document.getElementById('imgCam').src = data.gradcam_overlay_b64;
        document.getElementById('grid').style.display = 'grid';
      };
      reader.readAsDataURL(file);
    }
    document.getElementById('fileInput').addEventListener('change', e => handleFile(e.target.files[0]));
    document.getElementById('camInput').addEventListener('change', e => handleFile(e.target.files[0]));
  </script>
</body>
</html>"""


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    raw_b64 = req.image_base64.split(",", 1)[-1]
    try:
        img_bytes = base64.b64decode(raw_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Base64 inválido") from exc

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")

    proc_rgb = preprocess_image(img_bgr)
    x = np.ascontiguousarray(proc_rgb.transpose(2, 0, 1))[None, ...].astype(np.float32)

    sess, weights = _get_engine()
    input_name = sess.get_inputs()[0].name
    t0 = time.perf_counter()
    logit, conv_feat = sess.run(None, {input_name: x})
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    prob = float(1.0 / (1.0 + np.exp(-float(logit[0]))))
    is_cataract = prob >= 0.5

    base_rgb_u8 = np.clip((proc_rgb * _STD + _MEAN) * 255.0, 0, 255).astype(np.uint8)
    overlay_u8 = _compute_gradcam_overlay(base_rgb_u8, conv_feat, weights)

    return {
        "prediction": "CATARACT" if is_cataract else "NORMAL",
        "probability": round(prob, 4),
        "triage_message": (
            "SOSPECHA DE CATARATA — Canalizar a valoración oftalmológica"
            if is_cataract
            else "SIN SOSPECHA APARENTE (Normal)"
        ),
        "latency_ms": latency_ms,
        "preprocessed_b64": _encode_png_b64(base_rgb_u8),
        "gradcam_overlay_b64": _encode_png_b64(overlay_u8),
    }
