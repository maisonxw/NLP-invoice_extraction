import os
from pathlib import Path
from dotenv import load_dotenv

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

# API Keys
GEMINI_API_KEY = os.getenv("nhap api", "")

# Model Paths
LAYOUTLM_MODEL_PATH = PROJECT_ROOT / "train" / "output" / "model"

# Supported Formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
