from __future__ import annotations

from pathlib import Path

APP_NAME = "PixelBoost"
APP_DESCRIPTION = "AI-powered image upscaling for clearer, sharper visuals in a polished Streamlit experience."
SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "webp"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

REALESRGAN_MODEL_PATHS = {
    2: MODEL_DIR / "realesrgan" / "RealESRGAN_x2plus.pth",
    4: MODEL_DIR / "realesrgan" / "RealESRGAN_x4plus.pth",
}

OPENCV_MODEL_PATHS = {
    2: ("fsrcnn", MODEL_DIR / "opencv" / "FSRCNN_x2.pb"),
    4: ("edsr", MODEL_DIR / "opencv" / "EDSR_x4.pb"),
}

OPENCV_PROGRESSIVE_MODEL_PATH = MODEL_DIR / "opencv" / "FSRCNN_x2.pb"
FAST_4X_MAX_INPUT_PIXELS = 512 * 512
LARGE_IMAGE_WARNING_PIXELS = 1600 * 1600
