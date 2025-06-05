import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Model settings
MODEL_PATH = MODEL_DIR / "best_model.pth"
SEQUENCE_LENGTH = 10
DEVICE = "cuda" if os.environ.get("USE_CUDA", "false").lower() == "true" else "cpu"

# API settings
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "5000"))
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Image processing settings
IMAGE_SIZE = (224, 224)
TRANSFORM = {
    "resize": IMAGE_SIZE,
    "to_tensor": True
}

# Logging settings
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 