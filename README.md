# Sistema Inteligente para la Detección y Triaje de Cataratas

Proyecto de Redes Neuronales y Aprendizaje Profundo (**6BV1 • ESCOM – IPN**).
Herramienta de **triaje médico offline** basada en Redes Neuronales Convolucionales ligeras y Visión Explicable (**Grad-CAM**) para apoyar el tamizaje en dispositivos **Android** (no sustituye la valoración de un oftalmólogo).

## Estructura del Repositorio
```text
├── processed_images/          # Dataset Kaggle (nandanp6/cataract-image-dataset)
├── src/
│   ├── config.py              # Hiperparámetros y rutas centrales
│   ├── data/dataset.py        # Escaneo, deduplicación SHA-256 / dHash y DataLoader
│   ├── preprocessing/pipeline.py  # ROI pupilar, eliminación de reflejos, CLAHE, 224x224x3
│   ├── models/cnn.py          # EfficientNet-B0 y MobileNetV3-Small (GAP + Dropout 0.3 + Linear)
│   ├── evaluation/metrics.py  # Sensibilidad, Especificidad, Precisión (VPP), F2, AUC-ROC
│   ├── training/train_cv.py   # Validación cruzada estratificada por grupo (5-Fold)
│   ├── xai/gradcam.py         # Mapas de calor Grad-CAM + superposición (overlay)
│   └── export/export_onnx.py  # Exportación ONNX, cuantización INT8 y benchmark de latencia
├── android_app/               # Prototipo Android (Kotlin + ONNX Runtime Mobile + Grad-CAM)
├── artifacts/                 # Checkpoints, results_cv.json, gradcam/ y modelos ONNX
└── tests/test_pipeline.py     # Pruebas unitarias (preprocesamiento, CNN, Grad-CAM, ONNX)
```

## Resultados Parte 1: Validación Cruzada 5-Fold (Sin Fuga de Datos)
Tras eliminar `118` duplicados exactos en `train/` (`373` imágenes únicas: `225` `normal`, `148` `cataract`) y `10` duplicados + `59` coincidencias por hash perceptual en `test/` (`52` imágenes limpias):

| Modelo | Parámetros | Métrica | CV 5-Fold (Media ± DE) | Test Hold-out (52 imgs) |
| :--- | :---: | :--- | :---: | :---: |
| **EfficientNet-B0** | 4,008,829 | Sensibilidad (Recall) | **0.9455 ± 0.0351** | 0.8125 |
| | | Especificidad | 0.9556 ± 0.0344 | **0.9722** |
| | | Precisión (VPP) | **0.9363 ± 0.0435** | **0.9286** |
| | | F2-score | **0.9430 ± 0.0276** | 0.8333 |
| | | AUC-ROC | 0.9837 ± 0.0109 | 0.9844 |
| **MobileNetV3-Small** | **927,585** | Sensibilidad (Recall) | 0.9189 ± 0.0169 | **0.8750** |
| | | Especificidad | 0.9556 ± 0.0314 | 0.9444 |
| | | Precisión (VPP) | 0.9336 ± 0.0453 | 0.8750 |
| | | F2-score | 0.9213 ± 0.0145 | **0.8750** |
| | | AUC-ROC | **0.9845 ± 0.0079** | **0.9878** |

## Resultados Parte 2: XAI (Grad-CAM) y Exportación Móvil
- **Grad-CAM:** Se generaron 8 visualizaciones (`Input | Grad-CAM | Overlay`) en `artifacts/gradcam/`.
- **Benchmark ONNX (`mobilenet_v3_small`, Fold 5, 1 hilo CPU):**
  - **FP32 (`cataract_mobilenet_v3_small_fp32.onnx`):** `3.547 MB` (< 5 MB), latencia media `12.78 ms` / p95 `21.44 ms` (< 250 ms), `AUC-ROC = 0.9826`, `Especificidad = 0.9444`, `Precisión = 0.8571`.
  - **INT8 Post-Training (`cataract_mobilenet_v3_small_int8.onnx`):** `1.04–1.20 MB`, pero la cuantización post-entrenamiento sin QAT degrada las activaciones `HardSwish` de MobileNetV3 (`AUC-ROC = 0.41–0.65`). Dado que **FP32 ONNX (`3.55 MB`, `12.8 ms`) cumple holgadamente `< 5 MB` y `< 250 ms`** sin pérdida clínica, se integró en `android_app/app/src/main/assets/cataract_mobilenetv3.onnx`.

## Ejecución
### Python (Entrenamiento, XAI, Exportación y Tests)
```bash
pip install -r requirements.txt
python -m pytest -v
python -m src.training.train_cv --epochs 8
python -m src.xai.gradcam
python -m src.export.export_onnx
```

### Android Studio
1. Abre la carpeta `android_app/` en **Android Studio** (Gradle Sync descargará `onnxruntime-android:1.20.0`).
2. Verifica que `android_app/app/src/main/assets/` contenga `cataract_mobilenetv3.onnx` y `classifier_weights.json`.
3. Ejecuta en un dispositivo o emulador Android (`minSdk 24`). La app realiza el preprocesamiento, inferencia offline y mapa Grad-CAM localmente.
