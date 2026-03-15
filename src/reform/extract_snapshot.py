"""
ECG signal extraction — naive equal-slice approach.

Pipeline
--------
  pil_image
      └─ standardize()          resize to STANDARD_W × STANDARD_H, Otsu binarize
          └─ chop_leads()        trim header + cal pulse, equal-grid NumPy slice
              └─ extract_column_signal()  median Y of white pixels per column
                  └─ to_uv()             subtract mean, scale to µV
                      └─ resample_lead() interp1d → exactly LEAD_SAMPLES points
                          └─ highpass_filter()  remove baseline wander
                              └─ lowpass_filter()  remove HF noise
                                  └─ zero-pad into 5000-sample output vector

Unit grounding
--------------
Amplitude:  pixels_per_mv = pixels_per_mm × 10  (10 mm/mV standard)
            pixels_per_mm from detect_grid_spacing() on the original image
            (before resize, so grid lines are still visible).
            Fallback: STANDARD_W / 297 mm (A4 landscape) × 10 mm/mV.
            Baseline = mean(Y) of the extracted signal → centres at 0 mV.

Time:       3×4 layout: each column block = 2.5 s → 1250 samples at 500 Hz
            6×2 layout: each column block = 5.0 s → 2500 samples at 500 Hz
            Each lead is zero-padded into a 5000-sample (10 s) output vector
            at its correct temporal offset.

Lead layout
-----------
  3×4 (default):
    col 0   col 1   col 2   col 3
    I       AVR     V1      V4     ← row 0
    II      AVL     V2      V5     ← row 1
    III     AVF     V3      V6     ← row 2

  6×2:
    col 0   col 1
    I       V1     ← row 0
    II      V2     ← row 1
    III     V3     ← row 2
    AVR     V4     ← row 3
    AVL     V5     ← row 4
    AVF     V6     ← row 5
"""

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
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
OUTPUT_SAMPLES = 5000  # samples per output lead; 500 Hz × 10 s
SAMPLE_RATE_HZ = 500

LEAD_GRID_3x4 = [
    ["I", "AVR", "V1", "V4"],
    ["II", "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]

LEAD_GRID_6x2 = [
    ["I", "V1"],
    ["II", "V2"],
    ["III", "V3"],
    ["AVR", "V4"],
    ["AVL", "V5"],
    ["AVF", "V6"],
]

# Temporal slot of each lead inside the 5000-sample output vector
LEAD_SLOTS_3x4 = {
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

LEAD_SLOTS_6x2 = {
    "I": (0, 2500),
    "V1": (2500, 5000),
    "II": (0, 2500),
    "V2": (2500, 5000),
    "III": (0, 2500),
    "V3": (2500, 5000),
    "AVR": (0, 2500),
    "V4": (2500, 5000),
    "AVL": (0, 2500),
    "V5": (2500, 5000),
    "AVF": (0, 2500),
    "V6": (2500, 5000),
}

_LEAD_SAMPLES = {"3x4": 1250, "6x2": 2500}  # samples per lead at 500 Hz


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Standardize
# ──────────────────────────────────────────────────────────────────────────────


def standardize(pil_image: Image.Image, binarize: str = "color") -> np.ndarray:
    """
    Resize to STANDARD_W × STANDARD_H and binarize.

    Returns uint8 array (H × W): 255 = signal pixel, 0 = background.

    binarize : "color"    — HSV color filter (recommended; removes pink/red/blue
                            grid lines before thresholding the dark ECG trace)
               "canny"    — Canny edges + Hough line removal; useful for faded
                            prints where the grid and signal have similar intensity
               "otsu"     — global Otsu on grayscale (fast, but includes grid)
               "adaptive" — local Gaussian threshold (handles uneven lighting)
    """
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
# Step 2 — Chop into lead blocks
# ──────────────────────────────────────────────────────────────────────────────


def chop_leads(
    binary: np.ndarray,
    layout: str = "3x4",
    header_frac: float = 0.08,
    cal_frac: float = 0.03,
) -> dict[str, np.ndarray]:
    """
    Slice the binary image into one rectangular block per lead.

    No peak detection — pure NumPy array slicing into an equal grid.

    binary      : standardized binary image (H × W), 255 = signal
    layout      : "3x4" | "6x2"
    header_frac : fraction of total height to skip at the top as the patient
                  info / measurements header. Default 8%.
    cal_frac    : fraction of total width to skip at the left edge of each row
                  for the 1 mV calibration pulse printed before the signal.
                  At 25 mm/s the pulse is ~5 mm wide; at STANDARD_W ≈ 297 mm
                  that is ~1.7 %, so the default 3 % gives a small margin.

    Returns dict[lead_name → 2D uint8 block, 255 = signal].
    """
    grid = LEAD_GRID_3x4 if layout == "3x4" else LEAD_GRID_6x2
    n_rows = len(grid)
    n_cols = len(grid[0])

    H, W = binary.shape
    header_px = int(H * header_frac)
    cal_px = int(W * cal_frac)

    # Trim header from top, calibration pulse from left
    usable = binary[header_px:, cal_px:]
    h, w = usable.shape

    row_h = h // n_rows
    col_w = w // n_cols

    blocks: dict[str, np.ndarray] = {}
    for r, row_names in enumerate(grid):
        for c, name in enumerate(row_names):
            blocks[name] = usable[
                r * row_h : (r + 1) * row_h, c * col_w : (c + 1) * col_w
            ]

    return blocks


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Column-by-column signal extraction
# ──────────────────────────────────────────────────────────────────────────────


def extract_column_signal(block: np.ndarray) -> np.ndarray:
    """
    For each column in the block, collect the Y-coordinates of all white
    (signal) pixels and take their median to get one precise Y position.

    Using median instead of mean makes this robust to thick lines and any
    residual noise pixels.

    Empty columns (no signal pixels) are set to NaN and then filled by linear
    interpolation from neighbouring columns.

    Returns 1D float array of Y pixel positions, length = block width.
    Higher Y = lower on image = lower amplitude.
    """
    n_cols = block.shape[1]
    signal = np.full(n_cols, np.nan)

    for col in range(n_cols):
        white_rows = np.where(block[:, col] == 255)[0]
        if len(white_rows) > 0:
            signal[col] = float(np.median(white_rows))

    nan_mask = np.isnan(signal)
    if nan_mask.all():
        return np.zeros(n_cols)
    if nan_mask.any():
        idx = np.arange(n_cols)
        signal[nan_mask] = np.interp(idx[nan_mask], idx[~nan_mask], signal[~nan_mask])

    return signal


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 — Normalize and scale to µV
# ──────────────────────────────────────────────────────────────────────────────


def to_uv(signal: np.ndarray, pixels_per_mv: float) -> np.ndarray:
    """
    Centres the signal at 0 mV (subtract mean Y) and converts to µV.

    Y-axis is inverted (higher Y = lower on image = lower voltage):
      amplitude_mV = (mean_Y − Y) / pixels_per_mv
      amplitude_µV = amplitude_mV × 1000

    signal       : 1D Y-pixel array from extract_column_signal()
    pixels_per_mv: pixels per millivolt (from grid detection or fallback)
    """
    if pixels_per_mv == 0:
        return np.zeros_like(signal)
    baseline_y = float(np.mean(signal))
    return (baseline_y - signal) / pixels_per_mv


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 — Time interpolation
# ──────────────────────────────────────────────────────────────────────────────


def resample_lead(signal: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Stretch or compress signal to exactly target_samples points using
    scipy.interpolate.interp1d (linear).

    Example: a block with 800 raw pixel columns representing 2.5 s
             → resampled to exactly 1250 samples at 500 Hz.
    """
    if len(signal) == target_samples:
        return signal.copy()
    x_old = np.linspace(0.0, 1.0, len(signal))
    x_new = np.linspace(0.0, 1.0, target_samples)
    return interp1d(x_old, signal, kind="linear")(x_new)


# ──────────────────────────────────────────────────────────────────────────────
# Filters
# ──────────────────────────────────────────────────────────────────────────────


def highpass_filter(
    signal_uv: np.ndarray,
    cutoff_hz: float = 0.5,
    sample_rate: float = SAMPLE_RATE_HZ,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth high-pass filter.

    Removes slow baseline wander (e.g. breathing artefact, electrode drift)
    while preserving the ECG waveform shape.

    Clinical standard: 0.05 Hz preserves ST segments; 0.5 Hz removes wander
    more aggressively (may slightly distort ST). Default 0.5 Hz is a safe
    starting point for digitized paper ECGs where ST precision is secondary.

    signal_uv   : 1D µV array at sample_rate Hz
    cutoff_hz   : high-pass cutoff frequency (Hz)
    sample_rate : sample rate of the signal (default SAMPLE_RATE_HZ = 500)
    order       : Butterworth filter order (higher = sharper roll-off)
    """
    nyq = sample_rate / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="high")
    return filtfilt(b, a, signal_uv)


def lowpass_filter(
    signal_uv: np.ndarray,
    cutoff_hz: float = 40.0,
    sample_rate: float = SAMPLE_RATE_HZ,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth low-pass filter.

    Removes high-frequency noise (muscle artefact, electrical interference)
    while preserving the morphology of the ECG waveform.

    Clinical standards:
      40 Hz  — diagnostic quality (preserves fine detail, narrow QRS)
      150 Hz — full bandwidth (keeps all HF content, more noise)
      25 Hz  — aggressive smoothing (monitor quality, removes pacemaker spikes)

    Default 40 Hz is appropriate for most diagnostic use cases.

    signal_uv   : 1D µV array at sample_rate Hz
    cutoff_hz   : low-pass cutoff frequency (Hz)
    sample_rate : sample rate of the signal (default SAMPLE_RATE_HZ = 500)
    order       : Butterworth filter order
    """
    nyq = sample_rate / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal_uv)


# ──────────────────────────────────────────────────────────────────────────────
# Amplitude calibration — grid spacing detection
# ──────────────────────────────────────────────────────────────────────────────


def detect_grid_spacing(pil_image: Image.Image) -> float | None:
    """
    Estimates the ECG grid period in pixels/mm by autocorrelation of the
    image intensity profile. Run on the original image (before resizing) so
    the colored grid lines are maximally visible.

    Returns pixels_per_mm (on the original image), or None if detection fails.
    """
    img_np = np.array(pil_image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np

    estimates = []
    for profile in (gray.mean(axis=0), gray.mean(axis=1)):
        profile = profile.astype(float)
        profile -= profile.mean()

        autocorr = np.correlate(profile, profile, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]
        if autocorr[0] == 0:
            continue
        autocorr /= autocorr[0]

        peaks, _ = find_peaks(autocorr, distance=3, height=0.05)
        if len(peaks) < 3:
            continue

        period_px = float(np.median(np.diff(peaks[:10])))
        if period_px > 24:
            period_px /= 5.0
        if 2.0 <= period_px <= 25.0:
            estimates.append(period_px)

    return float(np.mean(estimates)) if estimates else None


# ──────────────────────────────────────────────────────────────────────────────
# Top-level pipeline
# ──────────────────────────────────────────────────────────────────────────────


def digitize(
    pil_image: Image.Image,
    layout: str = "3x4",
    dpi: int = 200,
    header_frac: float = 0.08,
    cal_frac: float = 0.03,
    hp_cutoff: float = 0.5,
    lp_cutoff: float = 40.0,
    binarize: str = "color",
) -> dict[str, np.ndarray]:
    """
    Full digitization pipeline: image → named µV lead signals.

    pil_image   : RGB PIL image of the ECG
    layout      : "3x4" (default) | "6x2"
    dpi         : scan DPI, used only to label the fallback (not computed from)
    header_frac : fraction of image height to skip as the patient info header
    cal_frac    : fraction of image width to skip as the 1 mV calibration pulse
                  at the left edge of each track (default 3 %)
    hp_cutoff   : high-pass filter cutoff in Hz (removes baseline wander).
                  Set to None to skip.
    lp_cutoff   : low-pass filter cutoff in Hz (removes HF noise).
                  Set to None to skip.
    binarize    : "color" (default) | "otsu" | "adaptive"

    Returns dict[lead_name → np.ndarray µV, length = OUTPUT_SAMPLES (5000)]
    Time axis: sample_index / SAMPLE_RATE_HZ → seconds.
    """
    slots = LEAD_SLOTS_3x4 if layout == "3x4" else LEAD_SLOTS_6x2
    lead_samples = _LEAD_SAMPLES[layout]

    # Amplitude calibration on original image (grid lines still visible)
    raw_ppm = detect_grid_spacing(pil_image)
    if raw_ppm is not None:
        pixels_per_mv = raw_ppm * (STANDARD_W / pil_image.width) * 10.0
    else:
        pixels_per_mv = (STANDARD_W / 297.0) * 10.0

    binary = standardize(pil_image, binarize=binarize)
    blocks = chop_leads(
        binary, layout=layout, header_frac=header_frac, cal_frac=cal_frac
    )

    leads: dict[str, np.ndarray] = {}
    for name, block in blocks.items():
        raw_y = extract_column_signal(block)
        signal_uv = to_uv(raw_y, pixels_per_mv)
        resampled = resample_lead(signal_uv, lead_samples)

        if hp_cutoff is not None:
            resampled = highpass_filter(resampled, cutoff_hz=hp_cutoff)
        if lp_cutoff is not None:
            resampled = lowpass_filter(resampled, cutoff_hz=lp_cutoff)

        out = np.zeros(OUTPUT_SAMPLES)
        start, end = slots[name]
        out[start:end] = resampled
        leads[name] = out

    return leads
