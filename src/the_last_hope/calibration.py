"""
Deterministic ECG calibration from image dimensions.

All ECG images in this dataset are generated at a fixed aspect ratio
(11 x 8.5 inches, US Letter) with standardised physical scaling:
  - 25 mm/s paper speed
  - 10 mm/mV amplitude gain

Because DPI = width / 11, every calibration parameter can be computed
directly from the pixel dimensions — no grid-line detection required.
"""

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physical constants (standard 12-lead ECG paper)
# ---------------------------------------------------------------------------
PAPER_WIDTH_INCHES = 11.0
PAPER_HEIGHT_INCHES = 8.5
MM_PER_MV = 10.0  # 10 mm per millivolt
MM_PER_SECOND = 25.0  # 25 mm per second
SAMPLE_RATE = 500  # Hz
SAMPLES_SHORT = 1250  # 2.5 s at 500 Hz
SAMPLES_FULL = 5000  # 10.0 s at 500 Hz

# ---------------------------------------------------------------------------
# Layout fractions — verified identical across all 3000 training images
# ---------------------------------------------------------------------------

# Column boundaries as fractions of image WIDTH (time axis, horizontal)
# 4 time columns: each spans 2.5 seconds
COL_STARTS = [0.054, 0.277, 0.501, 0.725]
COL_ENDS = [0.277, 0.501, 0.725, 0.948]

# Row lane boundaries as fractions of image HEIGHT (vertical)
# 3 signal rows + 1 rhythm row
ROW_STARTS = [0.334, 0.500, 0.667, 0.834]
ROW_ENDS = [0.500, 0.667, 0.834, 0.970]

# Lead assignment: LEAD_GRID[row][col] = lead_name
# Row index = spatial row on image (top to bottom)
# Col index = time column (left to right, each 2.5 s)
LEAD_GRID = [
    ["I", "aVR", "V1", "V4"],  # row 0
    ["II", "aVL", "V2", "V5"],  # row 1
    ["III", "aVF", "V3", "V6"],  # row 2
]

ALL_LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]

# Temporal offset for each lead within the 5000-sample output vector
LEAD_TEMPORAL_SLOTS = {
    "I": (0, 1250),
    "aVR": (1250, 2500),
    "V1": (2500, 3750),
    "V4": (3750, 5000),
    "II": (0, 1250),
    "aVL": (1250, 2500),
    "V2": (2500, 3750),
    "V5": (3750, 5000),
    "III": (0, 1250),
    "aVF": (1250, 2500),
    "V3": (2500, 3750),
    "V6": (3750, 5000),
}


@dataclass
class CalibrationParams:
    """All calibration values for a specific image size."""

    width: int
    height: int
    dpi: float
    pixels_per_mm: float
    pixels_per_mV: float
    # Absolute pixel boundaries for columns (time axis)
    col_x_starts: list[int]
    col_x_ends: list[int]
    # Absolute pixel boundaries for rows (lead lanes)
    row_y_starts: list[int]
    row_y_ends: list[int]


def compute_calibration(width: int, height: int) -> CalibrationParams:
    """Compute all calibration parameters from image pixel dimensions."""
    dpi = width / PAPER_WIDTH_INCHES
    pixels_per_mm = dpi / 25.4
    pixels_per_mV = pixels_per_mm * MM_PER_MV

    col_x_starts = [int(f * width) for f in COL_STARTS]
    col_x_ends = [int(f * width) for f in COL_ENDS]
    row_y_starts = [int(f * height) for f in ROW_STARTS]
    row_y_ends = [int(f * height) for f in ROW_ENDS]

    return CalibrationParams(
        width=width,
        height=height,
        dpi=dpi,
        pixels_per_mm=pixels_per_mm,
        pixels_per_mV=pixels_per_mV,
        col_x_starts=col_x_starts,
        col_x_ends=col_x_ends,
        row_y_starts=row_y_starts,
        row_y_ends=row_y_ends,
    )
