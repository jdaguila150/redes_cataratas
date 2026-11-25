"""Central configuration for the Cataract Triage project (Phase 1)."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "processed_images"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"

IMAGE_SIZE = 224
SEED = 42
N_FOLDS = 5
DROPOUT_P = 0.3

# Training hyperparameters (CPU-friendly defaults)
BATCH_SIZE = 16
EPOCHS = 12
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2

# ImageNet normalization statistics
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Class mapping: 0 = NORMAL, 1 = CATARACT
POSITIVE_FOLDERS = {"cataract", "immature", "mature"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
