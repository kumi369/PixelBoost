from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from pixelboost.config import SUPPORTED_FORMATS


def load_uploaded_image(uploaded_file) -> tuple[Image.Image, bytes]:
    filename = uploaded_file.name or "uploaded_image"
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in SUPPORTED_FORMATS:
        raise ValueError("Unsupported file format. Please upload JPG, JPEG, PNG, or WEBP.")

    file_bytes = uploaded_file.getvalue()
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        image = Image.open(BytesIO(file_bytes))
        image.load()
    except UnidentifiedImageError as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    return image, file_bytes


def get_image_details(image: Image.Image, image_bytes: bytes, filename: str) -> dict[str, str | int]:
    gcd = __import__("math").gcd(image.width, image.height)
    aspect_ratio = f"{image.width // gcd}:{image.height // gcd}" if gcd else "N/A"
    return {
        "name": filename,
        "width": image.width,
        "height": image.height,
        "aspect_ratio": aspect_ratio,
        "mode": image.mode,
        "format": image.format or "PNG",
        "size_kb": max(1, round(len(image_bytes) / 1024)),
    }


def split_alpha_if_present(image: Image.Image) -> tuple[Image.Image, Image.Image | None]:
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        rgb_image = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb_image.paste(rgba, mask=rgba.getchannel("A"))
        return rgb_image, rgba.getchannel("A")

    return image.convert("RGB"), None


def restore_alpha_if_needed(rgb_image: Image.Image, alpha_channel: Image.Image | None) -> Image.Image:
    if alpha_channel is None:
        return rgb_image

    resized_alpha = alpha_channel.resize(rgb_image.size, Image.Resampling.LANCZOS)
    rgba_image = rgb_image.convert("RGBA")
    rgba_image.putalpha(resized_alpha)
    return rgba_image


def pil_to_bgr_array(image: Image.Image) -> np.ndarray:
    rgb_image = image.convert("RGB")
    return np.array(rgb_image)[:, :, ::-1]


def bgr_array_to_pil(array: np.ndarray) -> Image.Image:
    rgb_array = array[:, :, ::-1]
    return Image.fromarray(rgb_array.astype(np.uint8))


def image_to_download_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    save_format = "PNG" if image.mode == "RGBA" else "PNG"
    image.save(buffer, format=save_format)
    return buffer.getvalue()


def build_download_filename(original_filename: str) -> str:
    stem = Path(original_filename).stem or "image"
    safe_stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in stem)
    safe_stem = safe_stem.strip("_") or "image"
    return f"pixelboost_{safe_stem}.png"
