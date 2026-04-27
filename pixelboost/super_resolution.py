from __future__ import annotations

from functools import lru_cache

import cv2
from PIL import Image

from pixelboost.config import OPENCV_MODEL_PATHS, REALESRGAN_MODEL_PATHS
from pixelboost.image_utils import (
    bgr_array_to_pil,
    pil_to_bgr_array,
    restore_alpha_if_needed,
    split_alpha_if_present,
)


class ModelConfigurationError(RuntimeError):
    """Raised when pretrained model files or dependencies are not configured correctly."""


class UpscalingError(RuntimeError):
    """Raised when image upscaling fails."""


class BaseUpscaler:
    backend_name: str

    def upscale(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError


class RealESRGANUpscaler(BaseUpscaler):
    def __init__(self, scale: int) -> None:
        self.scale = scale
        self.backend_name = f"Real-ESRGAN {scale}x"
        model_path = REALESRGAN_MODEL_PATHS.get(scale)
        if model_path is None or not model_path.exists():
            raise ModelConfigurationError(
                f"Missing Real-ESRGAN {scale}x weights at `{model_path}`."
            )

        try:
            import torch
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise ModelConfigurationError(
                "Real-ESRGAN dependencies are not installed. Install the packages in requirements.txt."
            ) from exc

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )

        gpu_available = torch.cuda.is_available()
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=str(model_path),
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=gpu_available,
            gpu_id=0 if gpu_available else None,
        )

    def upscale(self, image: Image.Image) -> Image.Image:
        rgb_image, alpha_channel = split_alpha_if_present(image)
        try:
            output_array, _ = self.upsampler.enhance(pil_to_bgr_array(rgb_image), outscale=self.scale)
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            raise UpscalingError(f"Real-ESRGAN processing failed: {exc}") from exc

        output_image = bgr_array_to_pil(output_array)
        return restore_alpha_if_needed(output_image, alpha_channel)


class OpenCVUpscaler(BaseUpscaler):
    def __init__(self, scale: int) -> None:
        self.scale = scale
        self.backend_name = f"OpenCV DNN SuperRes {scale}x"
        model_info = OPENCV_MODEL_PATHS.get(scale)
        if model_info is None:
            raise ModelConfigurationError(f"No OpenCV fallback model configured for {scale}x.")

        algorithm, model_path = model_info
        if not model_path.exists():
            raise ModelConfigurationError(f"Missing OpenCV fallback model at `{model_path}`.")

        if not hasattr(cv2, "dnn_superres"):
            raise ModelConfigurationError(
                "Your OpenCV build does not include dnn_superres. Install `opencv-contrib-python`."
            )

        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(str(model_path))
            self.sr.setModel(algorithm, scale)
        except cv2.error as exc:
            raise ModelConfigurationError(f"Failed to load OpenCV super-resolution model: {exc}") from exc

    def upscale(self, image: Image.Image) -> Image.Image:
        rgb_image, alpha_channel = split_alpha_if_present(image)

        try:
            output_array = self.sr.upsample(pil_to_bgr_array(rgb_image))
        except cv2.error as exc:
            raise UpscalingError(f"OpenCV super-resolution processing failed: {exc}") from exc

        output_image = bgr_array_to_pil(output_array)
        return restore_alpha_if_needed(output_image, alpha_channel)


@lru_cache(maxsize=4)
def get_upscaler(scale: int) -> BaseUpscaler:
    try:
        return RealESRGANUpscaler(scale)
    except ModelConfigurationError as real_esrgan_error:
        try:
            return OpenCVUpscaler(scale)
        except ModelConfigurationError as opencv_error:
            raise ModelConfigurationError(
                f"{real_esrgan_error} Fallback unavailable: {opencv_error}"
            ) from opencv_error


def upscale_image(image: Image.Image, scale: int) -> tuple[Image.Image, str]:
    upscaler = get_upscaler(scale)
    result = upscaler.upscale(image)
    return result, upscaler.backend_name
