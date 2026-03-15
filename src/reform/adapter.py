"""
Adapter: digitize() output → submission format.

Converts dict[lead_name → 5000-sample float64] from digitize()
into the competition format:

  Key   : {record_name}_{lead_name}  (e.g. "ecg_test_0001_I")
  Value : 1D np.float16 array, 5000 samples (10 s at 500 Hz)
          — NaN/Inf replaced with 0.0
  Units : millivolts (10 mm/mV, 25 mm/s)
"""

import numpy as np

from extract import LEAD_GRID, OUTPUT_SAMPLES

ALL_LEAD_NAMES = [name for row in LEAD_GRID for name in row]


def adapt(
    record_name: str,
    leads: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Convert one record's digitize() output to submission key/value pairs.

    Missing leads are zero-filled. NaN/Inf values are replaced with 0.
    """
    result = {}
    for lead_name in ALL_LEAD_NAMES:
        key = f"{record_name}_{lead_name}"
        if lead_name in leads:
            signal = np.nan_to_num(leads[lead_name], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            signal = np.zeros(OUTPUT_SAMPLES)
        result[key] = signal.astype(np.float16)
    return result
