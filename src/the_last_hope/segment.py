"""
Lead region slicing using fixed layout fractions.

The ECG layout is completely standardised — every image uses the same
proportional positioning.  This module slices the binary image into
13 rectangular blocks (12 short leads + 1 rhythm strip).
"""

import numpy as np

from calibration import CalibrationParams, LEAD_GRID


def segment_leads(
    binary: np.ndarray,
    cal: CalibrationParams,
) -> dict[str, np.ndarray]:
    """
    Slice binary image into one rectangular block per lead.

    Returns dict with 12 short-lead blocks keyed by lead name,
    plus "II_rhythm" for the full rhythm strip.
    """
    blocks: dict[str, np.ndarray] = {}

    for row_idx, row_leads in enumerate(LEAD_GRID):
        y0 = cal.row_y_starts[row_idx]
        y1 = cal.row_y_ends[row_idx]
        for col_idx, lead_name in enumerate(row_leads):
            x0 = cal.col_x_starts[col_idx]
            x1 = cal.col_x_ends[col_idx]
            blocks[lead_name] = binary[y0:y1, x0:x1]

    # Rhythm strip: spans all 4 time columns in row 3
    ry0 = cal.row_y_starts[3]
    ry1 = cal.row_y_ends[3]
    rx0 = cal.col_x_starts[0]
    rx1 = cal.col_x_ends[3]
    blocks["II_rhythm"] = binary[ry0:ry1, rx0:rx1]

    return blocks
