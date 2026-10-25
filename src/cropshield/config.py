import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"
DOCS_DIR = ROOT_DIR / "docs"
DETECTION_THRESHOLDS_PATH = DATA_DIR / "detection_thresholds.json"

MODEL_PATH = MODELS_DIR / "crop_model.keras"
TFLITE_MODEL_PATH = MODELS_DIR / "crop_model.tflite"
LABELS_PATH = MODELS_DIR / "class_labels.json"
METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"
TRAINING_HISTORY_PATH = ARTIFACTS_DIR / "training_history.csv"
DISEASE_KNOWLEDGE_PATH = DATA_DIR / "disease_knowledge.json"
LOCAL_EXPERT_CONTACTS_PATH = DATA_DIR / "local_expert_contacts.json"
PLANT_TIMELINE_IMAGES_DIR = DATA_DIR / "plant_timeline_images"

DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 12
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_SEED = 42

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_UPLOAD_SIZE_MB = 10
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_DIMENSION = 224


def get_env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_directories() -> None:
    for path in (DATA_DIR, MODELS_DIR, ARTIFACTS_DIR, REPORTS_DIR, DOCS_DIR):
        path.mkdir(parents=True, exist_ok=True)
