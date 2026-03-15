import os
from typing import Union
import numpy as np
from PIL import Image

from .errors import ImageLoadError


def load_image(image_source: Union[str, np.ndarray]) -> np.ndarray:
    """
    Load an image from a filepath or return a normalized copy of a numpy array.
    Ensures output is a 3-channel RGB uint8 numpy array with shape (H, W, 3).

    Args:
        image_source: File path to image or existing numpy array.

    Returns:
        np.ndarray: RGB image array of shape (H, W, 3) and dtype uint8.

    Raises:
        ImageLoadError: If the image cannot be read or processed.
    """
    if isinstance(image_source, str):
        if not os.path.exists(image_source):
            raise ImageLoadError(f"Image not found at path: {image_source}")

        try:
            # We use PIL here as it handles EXIF rotations better across platforms than raw cv2
            with Image.open(image_source) as img:
                img_data = np.array(img.convert("RGB"))
        except Exception as e:
            raise ImageLoadError(
                f"Failed to read image at {image_source}: {str(e)}"
            ) from e

    elif isinstance(image_source, np.ndarray):
        img_data = image_source.copy()

        # Handle grayscale (H, W) or (H, W, 1)
        if len(img_data.shape) == 2:
            img_data = np.stack((img_data,) * 3, axis=-1)
        elif len(img_data.shape) == 3 and img_data.shape[-1] == 1:
            img_data = np.concatenate([img_data] * 3, axis=-1)

        # Handle RGBA (H, W, 4)
        elif len(img_data.shape) == 3 and img_data.shape[-1] > 3:
            img_data = img_data[:, :, :3]

        if len(img_data.shape) != 3 or img_data.shape[-1] != 3:
            raise ImageLoadError(
                f"Expected image with 3 channels, got shape {img_data.shape}"
            )

        if img_data.dtype != np.uint8:
            # Basic normalization assumption if passed floats [0, 1]
            if img_data.max() <= 1.0:
                img_data = (img_data * 255).astype(np.uint8)
            else:
                img_data = img_data.astype(np.uint8)

    else:
        raise ImageLoadError(f"Unsupported image_source type: {type(image_source)}")

    return img_data
