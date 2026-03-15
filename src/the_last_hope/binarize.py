"""
Grid removal and signal isolation.

Three strategies:
  - Colored images (normal): HSV colour filter separates pink/red grid from
    the dark signal trace.
  - Colored images (temp-shifted): Grayscale darkness with expanded colour
    grid masking to handle orange/blue shifted grids.
  - BW images: adaptive thresholding + morphological grid-line removal.
"""

import cv2
import numpy as np

from calibration import CalibrationParams


# ---------------------------------------------------------------------------
# Detection: coloured vs black-and-white
# ---------------------------------------------------------------------------


def is_bw_image(rgb: np.ndarray) -> bool:
    """
    Return True if the image has no coloured grid (black-and-white / greyscale).

    Checks whether the grid region (background between signal traces) has
    meaningful colour saturation.  If saturation is uniformly low, the
    image is BW.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    # Look at pixels that are moderately bright (grid area, not signal)
    bright_mask = V > 100
    if bright_mask.sum() == 0:
        return True

    # Fraction of bright pixels that are also coloured (S > 30)
    coloured_frac = ((S > 30) & bright_mask).sum() / bright_mask.sum()
    return coloured_frac < 0.05


# ---------------------------------------------------------------------------
# Coloured-grid binarisation
# ---------------------------------------------------------------------------


def binarize_colored(rgb: np.ndarray) -> np.ndarray:
    """
    Isolate the dark ECG signal trace by filtering out coloured grid pixels.

    Works for both normal and color-temperature-shifted images by using
    grayscale darkness as the primary signal detector, excluding saturated
    bright pixels (which are grid).

    Returns uint8 binary image: 255 = signal, 0 = background/grid.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Grid pixels are colored (saturated) and relatively bright.
    # Signal is always darkest in grayscale, even when color-shifted.
    grid = (S > 30) & (V > 100)

    # Signal = dark in grayscale, not part of bright grid
    dark_thresh = min(max(np.percentile(gray, 3) * 1.5, 30), 100)
    signal_mask = (gray < dark_thresh) & ~grid

    return signal_mask.astype(np.uint8) * 255


# ---------------------------------------------------------------------------
# BW binarisation (adaptive threshold + morphological grid removal)
# ---------------------------------------------------------------------------


def binarize_bw(rgb: np.ndarray, cal: CalibrationParams) -> np.ndarray:
    """
    Binarise a black-and-white ECG image.

    Uses adaptive thresholding with a large block size to capture the
    signal trace as the locally darkest feature, followed by horizontal
    erosion/dilation to break thin vertical grid lines.

    Returns uint8 binary image: 255 = signal, 0 = background/grid.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Adaptive threshold with large block: captures the signal as the
    # locally darkest feature. Block size 91 and C=12 selected via
    # grid search on 20 BW training images (avg score 35.7).
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=91, C=12,
    )

    # Break thin vertical grid lines: erode then dilate horizontally.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    binary = cv2.erode(binary, h_kernel)
    binary = cv2.dilate(binary, h_kernel)

    return binary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def binarize(rgba: np.ndarray, cal: CalibrationParams) -> np.ndarray:
    """
    Auto-detect image type and binarise accordingly.

    Input: RGBA uint8 image (H, W, 3 or 4)
    Returns: uint8 binary image (H, W): 255 = signal, 0 = background
    """
    # Convert RGBA → RGB
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        rgb = rgba[:, :, :3]
    elif rgba.ndim == 3:
        rgb = rgba
    else:
        rgb = np.stack([rgba] * 3, axis=-1)

    bw = is_bw_image(rgb)
    if bw:
        binary = binarize_bw(rgb, cal)
    else:
        binary = binarize_colored(rgb)

    if not bw:
        # Clean up noise dots: remove isolated clusters smaller than 2x2
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)

        # Reconnect thin signal traces that may have been broken
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect_kernel)

    return binary
