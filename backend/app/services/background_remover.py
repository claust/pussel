"""Background removal service using rembg."""

import io
from typing import Optional, cast

from PIL import Image
from rembg import new_session, remove  # type: ignore[import-untyped]


class BackgroundRemover:
    """Removes background from puzzle piece images using rembg."""

    def __init__(self, model_name: str = "u2net") -> None:
        """Initialize with specified model.

        Args:
            model_name: rembg model to use. Options:
                - "u2net" (default, good balance of speed/quality)
                - "u2netp" (faster, smaller, lower quality)
                - "isnet-general-use" (better edge detection)
        """
        self._session = new_session(model_name)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        """Get the model name being used."""
        return self._model_name

    def remove_background(self, image_bytes: bytes) -> Image.Image:
        """Remove background from image.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)

        Returns:
            PIL Image with transparent background (RGBA)
        """
        input_image = Image.open(io.BytesIO(image_bytes))
        output_image = cast(Image.Image, remove(input_image, session=self._session))
        return output_image

    def remove_background_with_white(self, image_bytes: bytes) -> Image.Image:
        """Remove background and replace with white.

        This is useful for ML models that expect RGB input without transparency.

        Args:
            image_bytes: Raw image bytes (JPEG/PNG)

        Returns:
            PIL Image with white background (RGB)
        """
        rgba_image = self.remove_background(image_bytes)

        # Create white background
        white_bg = Image.new("RGB", rgba_image.size, (255, 255, 255))

        # Composite using alpha channel
        if rgba_image.mode == "RGBA":
            white_bg.paste(rgba_image, mask=rgba_image.split()[3])
        else:
            white_bg.paste(rgba_image)

        return white_bg


# Singleton instance
_background_remover: Optional[BackgroundRemover] = None


def get_background_remover(model_name: str = "u2net") -> BackgroundRemover:
    """Get or create singleton BackgroundRemover instance.

    Args:
        model_name: Model to use (only used on first call)

    Returns:
        The shared BackgroundRemover instance.
    """
    global _background_remover
    if _background_remover is None:
        _background_remover = BackgroundRemover(model_name=model_name)
    return _background_remover
