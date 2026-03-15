import cv2
import numpy as np
from PIL import Image


def color_filter_binarize(pil_image: Image.Image, dark_thresh: int = 80) -> Image.Image:
    """
    Binarizes an ECG image using HSV color space.

    ECG grids are typically pink/red or blue — these have high saturation and
    medium-high brightness. The signal trace is dark (low Value in HSV).

    Strategy:
      1. Build a mask for colored grid pixels (red/pink or blue hues, S > 40, V > 80)
      2. Build a mask for dark signal pixels (V < dark_thresh)
      3. Signal mask = dark AND NOT grid color

    Returns binary image: white = signal, black = background/grid.
    """
    img_np = np.array(pil_image)
    if len(img_np.shape) == 2:
        _, binary = cv2.threshold(img_np, dark_thresh, 255, cv2.THRESH_BINARY_INV)
        return Image.fromarray(binary)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # OpenCV HSV: H in [0, 179], S/V in [0, 255]
    # Pink/red: H 0–10 or 160–179 (wraps around red), with meaningful saturation
    red_pink = ((H <= 10) | (H >= 160)) & (S > 40) & (V > 80)
    # Blue: H roughly 100–130
    blue = (H >= 100) & (H <= 130) & (S > 40) & (V > 80)
    grid_mask = red_pink | blue

    # Signal: dark pixels that are not part of the colored grid
    signal_mask = (V < dark_thresh) & ~grid_mask

    binary = np.where(signal_mask, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def adaptive_binarize(
    pil_image: Image.Image, block_size: int = 35, C: int = 10
) -> Image.Image:
    """
    Binarizes an ECG image using cv2 Gaussian adaptive thresholding.

    Each pixel's threshold is computed from a local neighbourhood, so this
    handles uneven illumination (shadows, scan gradients) without needing
    color information.

    block_size: size of the local neighbourhood (must be odd)
    C: constant subtracted from the local mean — higher = less noise, fewer thin lines

    Returns binary image: white = signal, black = background.
    """
    img_np = np.array(pil_image)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    # THRESH_BINARY_INV: dark pixels (signal) → white, bright pixels → black
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        C,
    )
    return Image.fromarray(binary)


def otsu_binarize(pil_image: Image.Image, direction: str = "global"):
    """
    Performs Otsu thresholding.
    direction: "global" (whole image), "row" (row-by-row), or "col" (col-by-col)
    """
    # 1. Convert to grayscale numpy array
    img_np = np.array(pil_image)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    out_img = np.zeros_like(gray)

    if direction == "global":
        # Standard global Otsu
        _, out_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif direction == "row":
        # Apply Otsu independently across every row
        for i in range(gray.shape[0]):
            row_data = gray[i : i + 1, :]
            # Skip if the row is pure white or pure black
            if row_data.max() == row_data.min():
                out_img[i : i + 1, :] = row_data
            else:
                _, thresh_row = cv2.threshold(
                    row_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                out_img[i : i + 1, :] = thresh_row

    elif direction == "col":
        # Apply Otsu independently down every column
        for j in range(gray.shape[1]):
            col_data = gray[:, j : j + 1]
            if col_data.max() == col_data.min():
                out_img[:, j : j + 1] = col_data
            else:
                _, thresh_col = cv2.threshold(
                    col_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                out_img[:, j : j + 1] = thresh_col
    else:
        raise ValueError("Direction must be 'global', 'row', or 'col'")

    return Image.fromarray(out_img)


def canny_binarize(
    pil_image: Image.Image,
    dark_thresh: int = 80,
    hough_threshold: int = 80,
    min_line_length: int = 200,
    max_line_gap: int = 10,
) -> Image.Image:
    """
    Grid-aware binarization using Canny+Hough for grid detection.

    Strategy:
      1. Canny edge detect → HoughLinesP to find grid lines
      2. Build a grid mask from detected axis-aligned lines
      3. Threshold the original grayscale for dark (signal) pixels
      4. Subtract the grid mask → signal without grid

    Returns binary image: white = signal, black = background.
    """
    img_np = np.array(pil_image)
    gray = (
        cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np.copy()
    )

    # Step 1: find grid lines via Canny + Hough
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 100)

    grid_mask = np.zeros_like(gray)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 10 or angle > 170 or (80 < angle < 100):
                cv2.line(grid_mask, (x1, y1), (x2, y2), 255, thickness=5)

    # Step 2: threshold for dark pixels (signal + remaining grid ink)
    _, signal_mask = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)

    # Step 3: remove grid from signal
    signal_mask[grid_mask > 0] = 0

    return Image.fromarray(signal_mask)
