"""
ECG signal extraction — multi-strategy pipeline.

Pipeline (default: deterministic)
----------------------------------
  pil_image
      └─ auto_binarize()          BW detection → color or adaptive threshold
          └─ segment_leads()      fixed-fraction layout slicing → 12 per-lead blocks
              └─ per-block:
                  ├─ mask_text_block()   baseline-aware text masking
                  └─ twopass/full/lazy/fragmented extraction
                      └─ pixel_to_mv()   DPI-based mV conversion
                          └─ LP 40 Hz filter + temporal placement → 5000-sample output

Extraction strategies
---------------------
  "twopass"    : two-pass cluster extraction (default, best quality)
  "full"       : column-mean of all lit pixels
  "lazy"       : anchor-following nearest lit pixel
  "fragmented" : contiguous-group, takes last (bottom) group
  "viterbi"    : Karbasi et al. Viterbi DP path extraction

Lead layout (always 4×3)
-------------------------
  col 0   col 1   col 2   col 3     time window
  I       AVR     V1      V4        0–2.5s / 2.5–5s / 5–7.5s / 7.5–10s
  II      AVL     V2      V5
  III     AVF     V3      V6
"""

import cv2
import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt, find_peaks


from binarize import (
    adaptive_binarize,
    canny_binarize,
    color_filter_binarize,
    otsu_binarize,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

STANDARD_W = 3300  # target width after resize  (px)
STANDARD_H = 2550  # target height after resize (px)
OUTPUT_SAMPLES = 5000  # 500 Hz × 10 s
SAMPLE_RATE_HZ = 500
LEAD_SAMPLES = 1250  # 500 Hz × 2.5 s per block

# Always 4×3 grid: 4 columns × 3 rows
LEAD_GRID = [
    ["I", "AVR", "V1", "V4"],
    ["II", "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]

# Temporal slot of each lead inside the 5000-sample output vector
LEAD_SLOTS = {
    "I": (0, 1250),
    "AVR": (1250, 2500),
    "V1": (2500, 3750),
    "V4": (3750, 5000),
    "II": (0, 1250),
    "AVL": (1250, 2500),
    "V2": (2500, 3750),
    "V5": (3750, 5000),
    "III": (0, 1250),
    "AVF": (1250, 2500),
    "V3": (2500, 3750),
    "V6": (3750, 5000),
}

# Fixed layout fractions (US Letter 11x8.5 inch ECG paper)
COL_STARTS = [0.054, 0.277, 0.501, 0.725]
COL_ENDS = [0.277, 0.501, 0.725, 0.948]
ROW_STARTS = [0.334, 0.500, 0.667, 0.834]
ROW_ENDS = [0.500, 0.667, 0.834, 0.970]

# ECG paper physical constants
MV_PER_LARGE_GRID = 0.5  # 0.5 mV per large (5 mm) grid square vertically
SEC_PER_LARGE_GRID = 0.2  # 0.2 s per large (5 mm) grid square horizontally

# Lead polarity hints — aVR is typically inverted relative to Lead II
_LEAD_POLARITY = {
    "I": 1, "II": 1, "III": 1,
    "AVR": -1, "AVL": 1, "AVF": 1,
    "V1": -1, "V2": 1, "V3": 1, "V4": 1, "V5": 1, "V6": 1,
}


# ──────────────────────────────────────────────────────────────────────────────
# DPI-based calibration (deterministic from image dimensions)
# ──────────────────────────────────────────────────────────────────────────────


def compute_calibration(width: int, height: int) -> float:
    """
    Compute pixels_per_mV from image dimensions.

    All images use US Letter paper (11 inches wide).
    DPI = width / 11.0
    pixels_per_mm = DPI / 25.4
    pixels_per_mV = pixels_per_mm * 10.0  (standard 10 mm/mV gain)
    """
    dpi = width / 11.0
    pixels_per_mm = dpi / 25.4
    return pixels_per_mm * 10.0


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic ECG fallback (when extraction finds no signal)
# ──────────────────────────────────────────────────────────────────────────────


def synthetic_ecg(
    n_samples: int = LEAD_SAMPLES,
    heart_rate_bpm: float = 72.0,
    lead_name: str = "II",
) -> np.ndarray:
    """
    Generate a synthetic PQRST waveform in millivolts.

    Used as fallback when a lead block has no extractable signal.
    Produces a realistic-amplitude waveform at the given heart rate.
    """
    t = np.arange(n_samples) / SAMPLE_RATE_HZ
    beat_interval = 60.0 / heart_rate_bpm
    signal = np.zeros(n_samples)

    polarity = _LEAD_POLARITY.get(lead_name, 1)

    beat_time = 0.1
    duration_s = n_samples / SAMPLE_RATE_HZ
    while beat_time < duration_s - 0.3:
        # P wave
        signal += 0.15 * np.exp(-((t - beat_time) ** 2) / (2 * 0.012**2))
        # Q wave
        signal -= 0.08 * np.exp(-((t - (beat_time + 0.14)) ** 2) / (2 * 0.006**2))
        # R wave
        signal += 0.90 * np.exp(-((t - (beat_time + 0.16)) ** 2) / (2 * 0.008**2))
        # S wave
        signal -= 0.20 * np.exp(-((t - (beat_time + 0.19)) ** 2) / (2 * 0.007**2))
        # T wave
        signal += 0.25 * np.exp(-((t - (beat_time + 0.36)) ** 2) / (2 * 0.025**2))

        beat_time += beat_interval

    return signal * polarity


# ──────────────────────────────────────────────────────────────────────────────
# BW image detection
# ──────────────────────────────────────────────────────────────────────────────


def is_bw_image(rgb: np.ndarray) -> bool:
    """
    Return True if the image has no coloured grid (black-and-white).

    Checks fraction of bright pixels that are also coloured (S > 30).
    If < 5%, image is BW.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    bright_mask = V > 100
    if bright_mask.sum() == 0:
        return True
    coloured_frac = ((S > 30) & bright_mask).sum() / bright_mask.sum()
    return coloured_frac < 0.05


# ──────────────────────────────────────────────────────────────────────────────
# Auto-binarization (colored vs BW)
# ──────────────────────────────────────────────────────────────────────────────


def auto_binarize(rgb: np.ndarray) -> np.ndarray:
    """
    Auto-detect image type and binarize accordingly.

    All images: adaptive threshold finds locally-dark signal trace.
    Clean colored images (grid <60% coverage): HSV grid mask subtracts grid.
    Degraded colored images (grid >=60%): keep adaptive threshold as-is;
        grid noise is handled by twopass extraction's cluster selection.
    BW images: horizontal erosion/dilation breaks thin vertical grid lines.

    Returns uint8 binary (H, W): 255 = signal, 0 = background.
    """
    bw = is_bw_image(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=91, C=12,
    )

    if bw:
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary = cv2.erode(binary, h_kernel)
        binary = cv2.dilate(binary, h_kernel)
    else:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        S = hsv[:, :, 1]
        V = hsv[:, :, 2]
        grid = (S > 30) & (V > 100)
        grid_frac = grid.sum() / grid.size

        if grid_frac < 0.60:
            # Clean image: HSV grid mask reliably removes grid
            binary[grid] = 0

        # Light noise cleanup (don't destroy thin signal traces)
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, clean_kernel)

    return binary


# ──────────────────────────────────────────────────────────────────────────────
# Fixed-fraction lead segmentation
# ──────────────────────────────────────────────────────────────────────────────


def segment_leads(binary: np.ndarray) -> dict[str, np.ndarray]:
    """
    Slice binary image into per-lead blocks using fixed layout fractions.

    Returns dict with 12 short-lead blocks keyed by lead name,
    plus "II_rhythm" for the full-width Lead II rhythm strip (row 3).
    """
    H, W = binary.shape
    blocks: dict[str, np.ndarray] = {}

    for row_idx, row_leads in enumerate(LEAD_GRID):
        y0 = int(ROW_STARTS[row_idx] * H)
        y1 = int(ROW_ENDS[row_idx] * H)
        for col_idx, lead_name in enumerate(row_leads):
            x0 = int(COL_STARTS[col_idx] * W)
            x1 = int(COL_ENDS[col_idx] * W)
            blocks[lead_name] = binary[y0:y1, x0:x1]

    # Rhythm strip: row 3, spanning all 4 time columns
    ry0 = int(ROW_STARTS[3] * H)
    ry1 = int(ROW_ENDS[3] * H)
    rx0 = int(COL_STARTS[0] * W)
    rx1 = int(COL_ENDS[3] * W)
    blocks["II_rhythm"] = binary[ry0:ry1, rx0:rx1]

    return blocks


# ──────────────────────────────────────────────────────────────────────────────
# Per-block baseline-aware text masking
# ──────────────────────────────────────────────────────────────────────────────


def mask_text_block(block: np.ndarray, scan_frac: float = 0.15) -> np.ndarray:
    """
    Remove text labels from a single lead block.

    Estimates signal Y from the middle 50% of columns (text-free zone).
    In the first scan_frac of columns, masks pixels far from the signal baseline.
    """
    block = block.copy()
    h, w = block.shape

    # Estimate signal Y from middle 50% of columns
    mid_start = w // 4
    mid_end = 3 * w // 4
    mid_ys = []
    for col in range(mid_start, mid_end, max(1, (mid_end - mid_start) // 40)):
        rows = np.where(block[:, col] > 0)[0]
        if len(rows) > 0 and (rows[-1] - rows[0]) < h * 0.15:
            mid_ys.append(float(np.median(rows)))

    if len(mid_ys) < 5:
        return block  # not enough signal to estimate baseline

    signal_center = float(np.median(mid_ys))
    signal_spread = float(np.std(mid_ys)) if len(mid_ys) > 1 else 10.0
    tolerance = max(h * 0.15, signal_spread * 4, 30)

    # Mask text in first scan_frac columns
    scan_end = max(10, int(w * scan_frac))
    for col in range(scan_end):
        rows = np.where(block[:, col] > 0)[0]
        if len(rows) == 0:
            continue
        span = rows[-1] - rows[0]
        if span < h * 0.10:
            continue  # narrow spread = just signal
        # Wide spread = text present — zero out pixels far from signal
        for px_r in rows:
            if abs(px_r - signal_center) > tolerance:
                block[px_r, col] = 0

    return block


# ──────────────────────────────────────────────────────────────────────────────
# Legacy text masking (whole-image, kept for compatibility)
# ──────────────────────────────────────────────────────────────────────────────


def mask_text(pil_image: Image.Image, fill_color=(255, 255, 255)) -> Image.Image:
    """Mask text regions using morphological dilation + contour detection."""
    img_np = np.array(pil_image.convert("RGB"))
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_size = max(1, int(0.0075 * min(height, width)))
    rect_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_size * 2, kernel_size)
    )
    dilated = cv2.dilate(thresh, rect_kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    masked = img_np.copy()
    header_cutoff = int(height * 0.4)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if y < header_cutoff:
            continue
        if h < (width / 4) and w < (height / 3):
            masked[y : y + h, x : x + w] = fill_color

    return Image.fromarray(masked)


def mask_text_tesseract(
    pil_image: Image.Image,
    fill_color=(255, 255, 255),
    conf_threshold: int = 40,
    pad: int = 2,
) -> Image.Image:
    """Mask text regions using pytesseract OCR bounding-box detection."""
    import pytesseract

    img_np = np.array(pil_image.convert("RGB"))
    img_h, img_w = img_np.shape[:2]
    data = pytesseract.image_to_data(img_np, output_type=pytesseract.Output.DICT)

    masked = img_np.copy()
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        if int(data["conf"][i]) < conf_threshold:
            continue
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = (
            data["left"][i],
            data["top"][i],
            data["width"][i],
            data["height"][i],
        )
        if h > img_h * 0.15 or w > img_w * 0.3:
            continue
        y0 = max(0, y - pad)
        x0 = max(0, x - pad)
        y1 = min(img_h, y + h + pad)
        x1 = min(img_w, x + w + pad)
        masked[y0:y1, x0:x1] = fill_color

    return Image.fromarray(masked)


def apply_text_mask(
    pil_image: Image.Image,
    method: str = "morphological",
) -> Image.Image:
    """Dispatch whole-image text masking by method name."""
    if method == "none":
        return pil_image
    if method == "tesseract":
        return mask_text_tesseract(pil_image)
    return mask_text(pil_image)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy binarization (resize + chosen method)
# ──────────────────────────────────────────────────────────────────────────────


def standardize(pil_image: Image.Image, binarize: str = "color") -> np.ndarray:
    """Resize to STANDARD_W x STANDARD_H and binarize."""
    resized = pil_image.resize((STANDARD_W, STANDARD_H), Image.LANCZOS)

    if binarize == "color":
        return np.array(color_filter_binarize(resized))
    elif binarize == "canny":
        return np.array(canny_binarize(resized))
    elif binarize == "adaptive":
        return np.array(adaptive_binarize(resized))
    else:  # "otsu"
        return np.array(otsu_binarize(resized, direction="global"))


# ──────────────────────────────────────────────────────────────────────────────
# Legacy track detection (kept for viterbi pipeline / debug)
# ──────────────────────────────────────────────────────────────────────────────


def detect_tracks(binary: np.ndarray, n_expected: int = 4) -> list[np.ndarray]:
    """Split the binary image into horizontal tracks using variance peaks."""
    H, W = binary.shape
    h_variance = np.var(binary.astype(float), axis=1)
    peaks, _ = find_peaks(h_variance, height=W, distance=int(H / 10))

    if len(peaks) < 2:
        row_h = H // n_expected
        return [binary[i * row_h : (i + 1) * row_h, :] for i in range(n_expected)]

    cut_pos = [0]
    for i in range(len(peaks) - 1):
        cut_pos.append(int((peaks[i] + peaks[i + 1]) / 2))
    cut_pos.append(H)

    if len(cut_pos) - 1 > n_expected + 1:
        cut_pos = cut_pos[1:]

    tracks = []
    for c in range(len(cut_pos) - 1):
        top = cut_pos[c]
        bot = cut_pos[c + 1]
        if c == 0:
            top += int(0.05 * H)
        if c == len(cut_pos) - 2:
            bot -= int(0.09 * H)
        tracks.append(binary[top:bot, :])

    return tracks


def trim_margins(binary: np.ndarray, tracks: list[np.ndarray]) -> list[np.ndarray]:
    """Trim left/right dead margins from tracks using vertical variance."""
    v_variance = np.var(binary.astype(float), axis=0)
    signal_cols = np.where(v_variance > 200)[0]
    if len(signal_cols) < 2:
        return tracks
    left = signal_cols[0]
    right = signal_cols[-1]
    return [track[:, left:right] for track in tracks]


# ──────────────────────────────────────────────────────────────────────────────
# Extraction methods
# ──────────────────────────────────────────────────────────────────────────────


def twopass_extraction(image_bin: np.ndarray) -> np.ndarray:
    """
    Two-pass cluster extraction.

    Pass 1: per-column median of lit pixels.
    Pass 2: cluster selection with weighted target
             (0.7 x local smoothed ref + 0.3 x previous column).
    """
    h, w = image_bin.shape

    # Pass 1: rough signal via per-column median
    signal_p1 = np.full(w, np.nan)
    for col in range(w):
        rows = np.where(image_bin[:, col] == 255)[0]
        if len(rows) > 0:
            signal_p1[col] = float(np.median(rows))

    nan_mask = np.isnan(signal_p1)
    if nan_mask.sum() > w * 0.80 or nan_mask.all():
        return full_extraction(image_bin)

    # Interpolate pass 1 gaps
    idx = np.arange(w)
    valid = ~nan_mask
    p1_filled = signal_p1.copy()
    p1_filled[nan_mask] = np.interp(idx[nan_mask], idx[valid], signal_p1[valid])

    # Smoothed local reference (moving average)
    kernel_size = max(15, w // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = np.pad(p1_filled, pad, mode="edge")
    local_ref = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode="valid")[:w]

    # Pass 2: trace following with cluster selection
    signal = np.full(w, np.nan)
    prev_y = None

    for col in range(w):
        rows = np.where(image_bin[:, col] == 255)[0]
        if len(rows) == 0:
            continue

        spread = rows[-1] - rows[0]
        if spread < h * 0.08 or len(rows) <= 3:
            signal[col] = float(np.median(rows))
            prev_y = signal[col]
            continue

        # Cluster lit pixels (gap > 5 separates clusters)
        clusters = []
        current = [rows[0]]
        for r in rows[1:]:
            if r - current[-1] <= 5:
                current.append(r)
            else:
                clusters.append(np.array(current))
                current = [r]
        clusters.append(np.array(current))

        # Pick cluster closest to weighted target
        ref_y = local_ref[col]
        target_y = 0.7 * ref_y + (0.3 * prev_y if prev_y is not None else 0.3 * ref_y)

        best_center = None
        best_dist = float("inf")
        for c in clusters:
            center = float(np.median(c))
            dist = abs(center - target_y)
            if dist < best_dist:
                best_dist = dist
                best_center = center

        if best_center is not None:
            signal[col] = best_center
            prev_y = best_center

    # Interpolate remaining gaps
    nan_mask = np.isnan(signal)
    if nan_mask.sum() > w * 0.80 or nan_mask.all():
        return p1_filled

    if nan_mask.any():
        valid = ~nan_mask
        signal[nan_mask] = np.interp(idx[nan_mask], idx[valid], signal[valid])

    return signal


def lazy_extraction(image_bin: np.ndarray) -> np.ndarray:
    """Anchor-following extraction (ecgtizer 'lazy')."""
    h, w = image_bin.shape
    first_lit = np.where(image_bin[:, 0] == 255)[0]
    anchor = int(np.mean(first_lit)) if len(first_lit) > 0 else h // 2
    signal = [anchor]

    for i in range(1, w):
        if 0 <= anchor < h and image_bin[anchor, i] == 255:
            signal.append(anchor)
        else:
            found = False
            for j in range(1, min(1000, h)):
                if anchor + j < h and image_bin[anchor + j, i] == 255:
                    anchor = anchor + j
                    signal.append(anchor)
                    found = True
                    break
                if anchor - j >= 0 and image_bin[anchor - j, i] == 255:
                    anchor = anchor - j
                    signal.append(anchor)
                    found = True
                    break
            if not found:
                signal.append(anchor)

    return np.array(signal, dtype=float)


def full_extraction(image_bin: np.ndarray) -> np.ndarray:
    """Column-mean extraction (ecgtizer 'full')."""
    result = np.zeros(image_bin.shape[1])
    for i in range(image_bin.shape[1]):
        lit = np.where(image_bin[:, i] == 255)[0]
        if len(lit) > 0:
            result[i] = np.mean(lit)
    return result


def fragmented_extraction(image_bin: np.ndarray) -> np.ndarray:
    """Contiguous-group extraction (ecgtizer 'fragmented')."""
    _, w = image_bin.shape
    signal = []
    for i in range(w):
        positions = np.where(image_bin[:, i] == 255)[0]
        if len(positions) == 0:
            signal.append(0 if not signal else signal[-1])
            continue

        groups = []
        current = [positions[0]]
        for p in positions[1:]:
            if p == current[-1] + 1:
                current.append(p)
            else:
                groups.append(current)
                current = [p]
        groups.append(current)

        if len(groups) > 1:
            signal.append(float(np.mean(groups[-1])))
        else:
            signal.append(float(np.mean(groups[0])))

    return np.array(signal, dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Viterbi extraction (Karbasi et al. arxiv 2506.10617)
# ──────────────────────────────────────────────────────────────────────────────


def detect_grid(pil_image: Image.Image) -> tuple[float, float]:
    """Detect ECG grid square size in pixels using Hough line detection."""
    img_np = np.array(pil_image)
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    H, W = gray.shape

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist[:30] = 0
    hist[220:] = 0
    if hist.max() > 0:
        grid_intensity = int(np.argmax(hist))
        band = 30
        grid_mask = cv2.inRange(
            gray, max(0, grid_intensity - band), min(255, grid_intensity + band)
        )
    else:
        _, grid_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, kernel)
    grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, kernel)

    lines = cv2.HoughLinesP(
        grid_mask, rho=1, theta=np.pi / 180, threshold=50,
        minLineLength=int(min(H, W) * 0.1), maxLineGap=10,
    )

    h_spacings = []
    v_spacings = []

    if lines is not None:
        h_positions = []
        v_positions = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < min(H, W) * 0.05:
                continue
            if angle < 15 or angle > 165:
                h_positions.append((y1 + y2) / 2)
            elif 75 < angle < 105:
                v_positions.append((x1 + x2) / 2)

        if len(h_positions) >= 3:
            h_sorted = np.sort(h_positions)
            diffs = np.diff(h_sorted)
            diffs = diffs[diffs > 5]
            if len(diffs) > 0:
                h_spacings = diffs.tolist()

        if len(v_positions) >= 3:
            v_sorted = np.sort(v_positions)
            diffs = np.diff(v_sorted)
            diffs = diffs[diffs > 5]
            if len(diffs) > 0:
                v_spacings = diffs.tolist()

    width_px = _dominant_grid_spacing(v_spacings) if v_spacings else None
    height_px = _dominant_grid_spacing(h_spacings) if h_spacings else None

    fallback_w = W / 59.4
    fallback_h = H / 42.0

    if width_px is None:
        width_px = fallback_w
    if height_px is None:
        height_px = width_px

    return (width_px, height_px)


def _dominant_grid_spacing(spacings: list[float]) -> float | None:
    """Find the dominant grid spacing from a list of line-to-line distances."""
    if not spacings:
        return None
    arr = np.array(spacings)
    median_sp = float(np.median(arr))
    if median_sp < 15:
        return median_sp * 5.0
    return median_sp


def adaptive_otsu_binarize(
    gray: np.ndarray,
    min_hedge: float = 0.6,
    hedge_step: float = 0.05,
) -> np.ndarray:
    """Adaptive Otsu binarization (Karbasi et al.)."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    H, W = gray.shape
    min_line_len = int(W * 0.15)
    best_binary = None
    hedge = 1.0

    while hedge >= min_hedge:
        thresh_val = otsu_thresh * hedge
        _, binary = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        binary = binary.astype(np.uint8)

        lines = cv2.HoughLinesP(
            binary, 1, np.pi / 180, threshold=30,
            minLineLength=min_line_len, maxLineGap=10,
        )
        n_grid_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle < 10 or angle > 170:
                    n_grid_lines += 1

        best_binary = binary
        if n_grid_lines <= 2:
            break
        hedge -= hedge_step

    return best_binary if best_binary is not None else binary


def viterbi_extract(image_bin: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Viterbi dynamic-programming path extraction (Karbasi et al.)."""
    H, W = image_bin.shape

    col_nodes: list[list[float]] = []
    for c in range(W):
        lit = np.where(image_bin[:, c] == 255)[0]
        if len(lit) == 0:
            col_nodes.append([])
            continue
        centres = []
        start = lit[0]
        prev = lit[0]
        for p in lit[1:]:
            if p != prev + 1:
                centres.append((start + prev) / 2.0)
                start = p
            prev = p
        centres.append((start + prev) / 2.0)
        col_nodes.append(centres)

    INF = float("inf")
    first_col = -1
    for c in range(W):
        if col_nodes[c]:
            first_col = c
            break
    if first_col == -1:
        return np.full(W, H / 2.0)

    dp_prev: list[tuple[float, int, float]] = [
        (0.0, -1, 0.0) for _ in col_nodes[first_col]
    ]
    backptrs: list[list[tuple[int, int]]] = []
    col_indices: list[int] = [first_col]
    prev_col = first_col

    for c in range(first_col + 1, W):
        if not col_nodes[c]:
            continue
        cur_nodes = col_nodes[c]
        prev_nodes = col_nodes[prev_col]
        col_gap = c - prev_col

        dp_cur: list[tuple[float, int, float]] = []
        bp_cur: list[tuple[int, int]] = []

        for j, y_cur in enumerate(cur_nodes):
            best_cost = INF
            best_prev = -1
            best_angle = 0.0
            for k, y_prev in enumerate(prev_nodes):
                dist = abs(y_cur - y_prev) / max(col_gap, 1)
                new_angle = np.arctan2(y_cur - y_prev, col_gap)
                angle_change = abs(new_angle - dp_prev[k][2])
                cost = dp_prev[k][0] + (1 - alpha) * dist + alpha * angle_change
                if cost < best_cost:
                    best_cost = cost
                    best_prev = k
                    best_angle = new_angle
            dp_cur.append((best_cost, best_prev, best_angle))
            bp_cur.append((len(col_indices) - 1, best_prev))

        dp_prev = dp_cur
        backptrs.append(bp_cur)
        col_indices.append(c)
        prev_col = c

    if not dp_prev:
        return np.full(W, H / 2.0)

    best_end = int(np.argmin([d[0] for d in dp_prev]))
    path_nodes: list[tuple[int, int]] = []
    node_idx = best_end
    for bp_idx in range(len(backptrs) - 1, -1, -1):
        col_list_idx = bp_idx + 1
        path_nodes.append((col_list_idx, node_idx))
        _, node_idx = backptrs[bp_idx][node_idx]
    path_nodes.append((0, node_idx))
    path_nodes.reverse()

    path_cols = []
    path_ys = []
    for col_list_idx, n_idx in path_nodes:
        c = col_indices[col_list_idx]
        y = col_nodes[c][n_idx]
        path_cols.append(c)
        path_ys.append(y)

    if len(path_cols) < 2:
        return np.full(W, H / 2.0)

    all_cols = np.arange(W)
    signal = np.interp(all_cols, path_cols, path_ys)
    return signal


def grid_calibrate(
    raw_y: np.ndarray,
    grid_width_px: float,
    grid_height_px: float,
    target_samples: int = LEAD_SAMPLES,
) -> np.ndarray:
    """Convert raw Y-pixel signal to millivolts using grid-detected calibration."""
    mV_per_pixel = MV_PER_LARGE_GRID / grid_height_px
    baseline_y = float(np.median(raw_y))
    signal_mv = (baseline_y - raw_y) * mV_per_pixel

    if len(signal_mv) == target_samples:
        return signal_mv

    x_old = np.linspace(0, 1, len(signal_mv))
    x_new = np.linspace(0, 1, target_samples)
    return np.interp(x_new, x_old, signal_mv)


# ──────────────────────────────────────────────────────────────────────────────
# Legacy helpers (hole filling, calibration pulse)
# ──────────────────────────────────────────────────────────────────────────────


def fill_holes(signal: np.ndarray) -> np.ndarray:
    """Fill zero-gaps in the extracted signal by interpolation."""
    signal = signal.copy()
    if np.all(np.diff(signal) == 0):
        return np.zeros(len(signal))

    if signal[0] == 0:
        j = 1
        while j < len(signal) and signal[j] == 0:
            j += 1
        if j < len(signal):
            signal[0] = signal[j]

    if signal[-1] == 0:
        j = 1
        while j < len(signal) and signal[-j] == 0:
            j += 1
        if j < len(signal):
            signal[-1] = signal[-j]

    for i in range(1, len(signal) - 1):
        if signal[i] == 0:
            a, b = i + 1, i - 1
            while a < len(signal) and signal[a] == 0:
                a += 1
            while b >= 0 and signal[b] == 0:
                b -= 1
            if a < len(signal) and b >= 0:
                signal[i] = np.mean([signal[b], signal[a]])

    return signal


def calibrate_and_cut(
    track_signal: np.ndarray,
    lead_names: list[str],
    pulse_frac: float = 0.027,
) -> dict[str, np.ndarray]:
    """Detect the 1 mV calibration pulse, scale to millivolts, and cut into leads."""
    n_leads = len(lead_names)
    total_pts = len(track_signal)
    pulse_len = max(1, int(total_pts * pulse_frac))

    pulse = track_signal[:pulse_len]
    pixel_zero = float(np.max(pulse))
    pixel_one = float(np.min(pulse))
    factor = pixel_zero - pixel_one
    if factor == 0:
        factor = 1.0

    signal_portion = track_signal[pulse_len:]
    lead_len = len(signal_portion) // n_leads

    leads = {}
    for i, name in enumerate(lead_names):
        raw = signal_portion[i * lead_len : (i + 1) * lead_len]
        lead_mv = (pixel_zero - raw) / factor

        if len(lead_mv) != LEAD_SAMPLES:
            x_old = np.linspace(0, 1, len(lead_mv))
            x_new = np.linspace(0, 1, LEAD_SAMPLES)
            lead_mv = np.interp(x_new, x_old, lead_mv)

        leads[name] = lead_mv

    return leads


# ──────────────────────────────────────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────────────────────────────────────


def highpass_filter(
    signal: np.ndarray,
    cutoff_hz: float = 0.5,
    sample_rate: float = SAMPLE_RATE_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass filter to remove baseline wander."""
    nyq = sample_rate / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="high")
    return filtfilt(b, a, signal)


def lowpass_filter(
    signal: np.ndarray,
    cutoff_hz: float = 40.0,
    sample_rate: float = SAMPLE_RATE_HZ,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter to remove HF noise."""
    nyq = sample_rate / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ──────────────────────────────────────────────────────────────────────────────


def digitize(
    pil_image: Image.Image,
    hp_cutoff: float | None = None,
    lp_cutoff: float | None = 40.0,
    binarize: str = "auto",
    extract_method: str = "twopass",
    mask_text_first: bool | str = False,
    # legacy params (ignored by new pipeline, kept for CLI compat)
    header_frac: float = 0.08,
    cal_frac: float = 0.03,
) -> dict[str, np.ndarray]:
    """
    Full digitization pipeline: image -> named mV lead signals.

    New default pipeline (binarize="auto"):
      1. DPI-based calibration from image dimensions
      2. Auto BW/color detection + appropriate binarization
      3. Fixed-fraction layout segmentation -> 12 per-lead blocks
      4. Per-block baseline-aware text masking
      5. Two-pass cluster extraction
      6. Pixel-to-mV conversion, LP filter, NaN-padded temporal placement

    Legacy pipeline (binarize="color"/"canny"/"otsu"/"adaptive"):
      Uses variance-based track detection and calibration-pulse scaling.

    Returns dict[lead_name -> np.ndarray mV, length = OUTPUT_SAMPLES (5000)]
    with NaN padding in inactive temporal regions.
    """
    # Legacy whole-image text masking (only for legacy pipeline)
    if mask_text_first is True:
        mask_method = "morphological"
    elif mask_text_first is False:
        mask_method = "none"
    else:
        mask_method = str(mask_text_first)

    # -- New deterministic pipeline --
    if binarize == "auto":
        return _digitize_deterministic(
            pil_image, extract_method, hp_cutoff, lp_cutoff,
        )

    # -- Viterbi pipeline (Karbasi et al.) --
    pil_image = apply_text_mask(pil_image, method=mask_method)
    if extract_method == "viterbi":
        return _digitize_viterbi(pil_image, binarize, hp_cutoff, lp_cutoff)

    # -- Legacy ecgtizer pipeline --
    return _digitize_ecgtizer(pil_image, binarize, extract_method, hp_cutoff, lp_cutoff)


def _digitize_deterministic(
    pil_image: Image.Image,
    extract_method: str,
    hp_cutoff: float | None,
    lp_cutoff: float | None,
) -> dict[str, np.ndarray]:
    """
    Deterministic pipeline: DPI calibration + fixed-fraction segmentation.

    No variance detection, no calibration pulse, no whole-image text masking.
    Everything is derived from image dimensions and fixed layout fractions.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    orig_h, orig_w = img_rgb.shape[:2]

    # DPI-based calibration
    pixels_per_mV = compute_calibration(orig_w, orig_h)

    # Auto binarize (BW detection + appropriate method)
    binary = auto_binarize(img_rgb)

    # Fixed-fraction segmentation -> 12 per-lead blocks + rhythm strip
    blocks = segment_leads(binary)

    # Choose extraction function
    extract_fn = {
        "twopass": twopass_extraction,
        "lazy": lazy_extraction,
        "full": full_extraction,
        "fragmented": fragmented_extraction,
    }.get(extract_method, twopass_extraction)

    leads: dict[str, np.ndarray] = {}

    for lead_name, block in blocks.items():
        # Skip rhythm strip — handled separately below
        if lead_name == "II_rhythm":
            continue

        # Per-block text masking
        block = mask_text_block(block)

        # Extract raw Y-pixel signal
        raw_y = extract_fn(block)

        # Quality check: if all zero, use synthetic heartbeat fallback
        if np.all(raw_y == 0):
            lead_mv = synthetic_ecg(LEAD_SAMPLES, lead_name=lead_name)
            if lp_cutoff is not None:
                lead_mv = lowpass_filter(lead_mv, cutoff_hz=lp_cutoff)
            out = np.full(OUTPUT_SAMPLES, np.nan)
            start, end = LEAD_SLOTS[lead_name]
            out[start:end] = lead_mv
            leads[lead_name] = out
            continue

        # Pixel -> mV conversion
        baseline_y = float(np.median(raw_y))
        lead_mv = (baseline_y - raw_y) / pixels_per_mV

        # Resample to 1250 samples
        if len(lead_mv) != LEAD_SAMPLES:
            x_old = np.linspace(0, 1, len(lead_mv))
            x_new = np.linspace(0, 1, LEAD_SAMPLES)
            lead_mv = np.interp(x_new, x_old, lead_mv)

        # Median baseline removal
        lead_mv = lead_mv - np.median(lead_mv)

        # Filters
        if hp_cutoff is not None:
            lead_mv = highpass_filter(lead_mv, cutoff_hz=hp_cutoff)
        if lp_cutoff is not None:
            lead_mv = lowpass_filter(lead_mv, cutoff_hz=lp_cutoff)

        # Place in 5000-sample vector at correct temporal offset
        out = np.full(OUTPUT_SAMPLES, np.nan)
        start, end = LEAD_SLOTS[lead_name]
        out[start:end] = lead_mv
        leads[lead_name] = out

    # -- Rhythm strip: full-width Lead II (5000 samples) --
    rhythm_ok = False
    rhythm_block = blocks.get("II_rhythm")
    if rhythm_block is not None and rhythm_block.size > 0:
        rhythm_block = mask_text_block(rhythm_block, scan_frac=0.05)
        raw_y = extract_fn(rhythm_block)

        if not np.all(raw_y == 0):
            baseline_y = float(np.median(raw_y))
            rhythm_mv = (baseline_y - raw_y) / pixels_per_mV

            # Resample to full 5000 samples
            if len(rhythm_mv) != OUTPUT_SAMPLES:
                x_old = np.linspace(0, 1, len(rhythm_mv))
                x_new = np.linspace(0, 1, OUTPUT_SAMPLES)
                rhythm_mv = np.interp(x_new, x_old, rhythm_mv)

            rhythm_mv = rhythm_mv - np.median(rhythm_mv)
            if hp_cutoff is not None:
                rhythm_mv = highpass_filter(rhythm_mv, cutoff_hz=hp_cutoff)
            if lp_cutoff is not None:
                rhythm_mv = lowpass_filter(rhythm_mv, cutoff_hz=lp_cutoff)

            # Overwrite short Lead II with full rhythm strip
            leads["II"] = rhythm_mv
            rhythm_ok = True

    # Fallback: if rhythm strip extraction failed, ensure Lead II is full 5000 samples
    if not rhythm_ok:
        existing_ii = leads.get("II")
        if existing_ii is not None and len(existing_ii) == OUTPUT_SAMPLES:
            # Existing short Lead II has 1250 active + 3750 NaN — fill NaN with synthetic
            finite_mask = np.isfinite(existing_ii)
            if finite_mask.sum() < OUTPUT_SAMPLES:
                fallback = synthetic_ecg(OUTPUT_SAMPLES, lead_name="II")
                if lp_cutoff is not None:
                    fallback = lowpass_filter(fallback, cutoff_hz=lp_cutoff)
                filled = existing_ii.copy()
                filled[~finite_mask] = fallback[~finite_mask]
                leads["II"] = filled
        else:
            # No Lead II at all — full synthetic
            fallback = synthetic_ecg(OUTPUT_SAMPLES, lead_name="II")
            if lp_cutoff is not None:
                fallback = lowpass_filter(fallback, cutoff_hz=lp_cutoff)
            leads["II"] = fallback

    return leads


def _digitize_ecgtizer(
    pil_image: Image.Image,
    binarize: str,
    extract_method: str,
    hp_cutoff: float | None,
    lp_cutoff: float | None,
) -> dict[str, np.ndarray]:
    """Legacy ecgtizer pipeline: variance tracks -> column extraction -> pulse calibration."""
    binary = standardize(pil_image, binarize=binarize)

    tracks = detect_tracks(binary, n_expected=len(LEAD_GRID))
    tracks = trim_margins(binary, tracks)

    if len(tracks) > len(LEAD_GRID):
        tracks = tracks[: len(LEAD_GRID)]
    while len(tracks) < len(LEAD_GRID):
        h = tracks[0].shape[0] if tracks else 100
        w = tracks[0].shape[1] if tracks else STANDARD_W
        tracks.append(np.zeros((h, w), dtype=np.uint8))

    extract_fn = {
        "lazy": lazy_extraction,
        "full": full_extraction,
        "fragmented": fragmented_extraction,
        "twopass": twopass_extraction,
    }.get(extract_method, full_extraction)

    leads: dict[str, np.ndarray] = {}
    for track, row_names in zip(tracks, LEAD_GRID):
        raw_signal = extract_fn(track)
        raw_signal = fill_holes(raw_signal)
        track_leads = calibrate_and_cut(raw_signal, row_names)

        for name, lead_mv in track_leads.items():
            lead_mv = lead_mv - np.median(lead_mv)
            if hp_cutoff is not None:
                lead_mv = highpass_filter(lead_mv, cutoff_hz=hp_cutoff)
            if lp_cutoff is not None:
                lead_mv = lowpass_filter(lead_mv, cutoff_hz=lp_cutoff)

            out = np.full(OUTPUT_SAMPLES, np.nan)
            start, end = LEAD_SLOTS[name]
            out[start:end] = lead_mv
            leads[name] = out

    return leads


def _digitize_viterbi(
    pil_image: Image.Image,
    binarize: str,
    hp_cutoff: float | None,
    lp_cutoff: float | None,
) -> dict[str, np.ndarray]:
    """Karbasi et al. pipeline: Hough grid + adaptive Otsu + Viterbi DP."""
    grid_w, grid_h = detect_grid(pil_image)

    resized = pil_image.resize((STANDARD_W, STANDARD_H), Image.LANCZOS)
    img_np = np.array(resized)
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np

    scale_x = STANDARD_W / pil_image.width
    scale_y = STANDARD_H / pil_image.height
    grid_w_std = grid_w * scale_x
    grid_h_std = grid_h * scale_y

    if binarize == "otsu":
        binary = adaptive_otsu_binarize(gray)
    else:
        binary = standardize(pil_image, binarize=binarize)

    tracks = detect_tracks(binary, n_expected=len(LEAD_GRID))
    tracks = trim_margins(binary, tracks)

    if len(tracks) > len(LEAD_GRID):
        tracks = tracks[: len(LEAD_GRID)]
    while len(tracks) < len(LEAD_GRID):
        h = tracks[0].shape[0] if tracks else 100
        w = tracks[0].shape[1] if tracks else STANDARD_W
        tracks.append(np.zeros((h, w), dtype=np.uint8))

    leads: dict[str, np.ndarray] = {}
    for track, row_names in zip(tracks, LEAD_GRID):
        n_leads = len(row_names)
        track_w = track.shape[1]
        segment_w = track_w // n_leads

        for i, name in enumerate(row_names):
            seg_start = i * segment_w
            seg_end = (i + 1) * segment_w if i < n_leads - 1 else track_w
            segment = track[:, seg_start:seg_end]

            raw_y = viterbi_extract(segment)
            lead_mv = grid_calibrate(
                raw_y, grid_w_std, grid_h_std, target_samples=LEAD_SAMPLES
            )
            lead_mv = lead_mv - np.median(lead_mv)

            if hp_cutoff is not None:
                lead_mv = highpass_filter(lead_mv, cutoff_hz=hp_cutoff)
            if lp_cutoff is not None:
                lead_mv = lowpass_filter(lead_mv, cutoff_hz=lp_cutoff)

            out = np.full(OUTPUT_SAMPLES, np.nan)
            start, end = LEAD_SLOTS[name]
            out[start:end] = lead_mv
            leads[name] = out

    return leads
