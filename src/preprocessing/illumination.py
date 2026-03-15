import cv2
import numpy as np

from .models import PreprocessingConfig
from .errors import IlluminationError, IlluminationConfigError


def apply_illumination_normalization(
    image: np.ndarray, config: PreprocessingConfig
) -> np.ndarray:
    """
    Applies illumination normalization to an RGB image based on the provided configuration.

    Args:
        image: Original RGB numpy array of shape (H, W, 3) and dtype uint8.
        config: Configuration dictating which method and parameters to use.

    Returns:
        A new RGB numpy array (H, W, 3) of dtype uint8 with illumination normalization applied.
    """
    if not config.enable_illumination_normalization:
        return image

    method = config.illumination_method.lower().strip()

    try:
        if method == "morphology":
            return _apply_baseline_morphology(image, config)
        elif method == "clahe":
            return _apply_advanced_clahe(image, config)
        else:
            raise IlluminationConfigError(
                f"Unknown illumination method '{method}'. Valid options are: 'morphology', 'clahe'."
            )
    except Exception as e:
        if isinstance(e, IlluminationConfigError):
            raise
        raise IlluminationError(
            f"Failed to apply illumination normalization: {str(e)}"
        ) from e


def _apply_baseline_morphology(
    image: np.ndarray, config: PreprocessingConfig
) -> np.ndarray:
    """
    Baseline method: Morphological white tophat.
    Best for removing crease shadows and broad lighting variations.
    """
    kernel_size = config.illumination_morph_kernel_size
    if kernel_size < 3 or kernel_size % 2 == 0:
        # Standardize kernel size to be a logical odd number
        kernel_size = max(3, kernel_size + 1 if kernel_size % 2 == 0 else kernel_size)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    if len(image.shape) == 2:
        # Processing a localized 1D L-channel calculation array
        channels = [image]
    else:
        # Process each color channel independently to preserve standard RGB input formatting
        channels = cv2.split(image)

    normalized_channels = []

    for ch in channels:
        # Estimate the background illumination by applying a morphological closing.
        # Closing removes dark features smaller than the kernel (like text, gridlines, ECG leads),
        # but leaves large dark features (like broad shadows and uneven lighting).
        bg = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, kernel)

        # Divide the original image by the estimated background to normalize illumination.
        # This isolates the reflectance (the ink/traces) from the global lighting.
        bg_float = bg.astype(np.float32)
        bg_float[bg_float == 0] = 1.0  # Prevent division by zero

        normalized_ch = (ch.astype(np.float32) / bg_float) * 255.0
        normalized_ch = np.clip(normalized_ch, 0, 255).astype(np.uint8)
        normalized_channels.append(normalized_ch)

    normalized = (
        cv2.merge(normalized_channels)
        if len(normalized_channels) > 1
        else normalized_channels[0]
    )
    return normalized


def _apply_advanced_clahe(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """
    Advanced method: Morphology + CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Provides robust local contrast enhancement in heavily crumpled dark regions.
    """
    # First remove global broad biases
    debiased_image = _apply_baseline_morphology(image, config)

    # Check if we are processing a 1D structural channel or standard 3D RGB
    is_grayscale = len(debiased_image.shape) == 2

    if is_grayscale:
        l_channel = debiased_image
    else:
        # Convert to LAB to process luminosity independently of color
        lab = cv2.cvtColor(debiased_image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

    # Apply CLAHE to the Lightness/Grayscale channel
    tile_size = config.illumination_clahe_tile_size
    clahe = cv2.createCLAHE(
        clipLimit=config.illumination_clahe_clip_limit,
        tileGridSize=(tile_size, tile_size),
    )
    l_clahe = clahe.apply(l_channel)

    if is_grayscale:
        return l_clahe

    # Merge back and convert to RGB
    lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return rgb_clahe
