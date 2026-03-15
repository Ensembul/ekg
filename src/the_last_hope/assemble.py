"""
Signal assembly: pixel positions -> calibrated mV signals at 500 Hz.

Handles unit conversion, resampling, baseline correction, and filtering.

Output format: every lead is a 5000-sample vector at 500 Hz.
  - Short leads (2.5s visible): 1250 real samples placed at the correct
    temporal offset, rest filled with NaN.
  - Lead II (rhythm strip): full 5000 samples.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from calibration import (
    CalibrationParams,
    LEAD_GRID,
    LEAD_TEMPORAL_SLOTS,
    SAMPLES_SHORT,
    SAMPLES_FULL,
    SAMPLE_RATE,
    ALL_LEAD_NAMES,
)
from extract import extract_signal


def pixels_to_mv(signal_y: np.ndarray, cal: CalibrationParams) -> np.ndarray:
    """
    Convert Y-pixel positions to millivolts.

    Higher Y on image = lower on paper = lower voltage, so we invert.
    Baseline is estimated as the median Y position (robust to QRS spikes).
    """
    if cal.pixels_per_mV == 0:
        return np.zeros_like(signal_y)

    baseline_y = float(np.median(signal_y))
    amplitude_px = baseline_y - signal_y  # invert: higher Y -> lower voltage
    return amplitude_px / cal.pixels_per_mV


def resample_lead(signal: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Resample signal to exactly target_samples using cubic interpolation.
    """
    n = len(signal)
    if n == target_samples:
        return signal.copy()
    if n < 4:
        return np.zeros(target_samples)

    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, target_samples)
    return interp1d(x_old, signal, kind="cubic")(x_new)


def bandpass_filter(
    signal_mv: np.ndarray,
    hp_cutoff: float = 0.5,
    lp_cutoff: float = 40.0,
    sample_rate: float = SAMPLE_RATE,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.

    hp_cutoff: removes baseline wander (0.5 Hz)
    lp_cutoff: removes HF noise (40 Hz)
    """
    nyq = sample_rate / 2.0

    # Need sufficient samples for the filter
    if len(signal_mv) < 3 * order + 1:
        return signal_mv

    if hp_cutoff is not None and hp_cutoff > 0:
        b, a = butter(order, hp_cutoff / nyq, btype="high")
        padlen = min(3 * max(len(a), len(b)), len(signal_mv) - 1)
        if padlen > 0:
            signal_mv = filtfilt(b, a, signal_mv, padlen=padlen)

    if lp_cutoff is not None and lp_cutoff > 0:
        b, a = butter(order, lp_cutoff / nyq, btype="low")
        padlen = min(3 * max(len(a), len(b)), len(signal_mv) - 1)
        if padlen > 0:
            signal_mv = filtfilt(b, a, signal_mv, padlen=padlen)

    return signal_mv


def assemble_lead(
    block: np.ndarray,
    lead_name: str,
    cal: CalibrationParams,
    is_rhythm: bool = False,
) -> np.ndarray:
    """
    Extract, calibrate, resample, and filter a single lead.

    Returns the signal as a 1D float64 array:
      - Short leads: 1250 samples
      - Rhythm strip: 5000 samples
    """
    target = SAMPLES_FULL if is_rhythm else SAMPLES_SHORT

    # Extract raw Y-pixel positions
    raw_y = extract_signal(block)
    if np.all(raw_y == 0):
        return np.zeros(target)

    # Convert to millivolts
    signal_mv = pixels_to_mv(raw_y, cal)

    # Resample to target sample count
    resampled = resample_lead(signal_mv, target)

    # Apply lowpass filter only (HP filter distorts short 2.5s leads)
    resampled = bandpass_filter(resampled, hp_cutoff=None, lp_cutoff=40.0)

    return resampled


def assemble_record(
    blocks: dict[str, np.ndarray],
    cal: CalibrationParams,
) -> dict[str, np.ndarray]:
    """
    Assemble all 12 leads for one ECG record.

    Returns: dict mapping lead name -> 5000-sample float64 array.
      - Short leads: 1250 real samples placed at correct temporal offset,
        rest filled with NaN.
      - Lead II: full 5000 samples from the rhythm strip.
    """
    leads: dict[str, np.ndarray] = {}

    for row_leads in LEAD_GRID:
        for lead_name in row_leads:
            if lead_name == "II":
                continue
            block = blocks.get(lead_name)
            if block is None or block.size == 0:
                short_signal = np.zeros(SAMPLES_SHORT)
            else:
                short_signal = assemble_lead(block, lead_name, cal)

            # Place at correct temporal offset in 5000-sample vector
            t0, t1 = LEAD_TEMPORAL_SLOTS[lead_name]
            full = np.full(SAMPLES_FULL, np.nan)
            full[t0:t1] = short_signal
            leads[lead_name] = full

    # Lead II from rhythm strip (full 5000 samples)
    rhythm_block = blocks.get("II_rhythm")
    if rhythm_block is not None and rhythm_block.size > 0:
        leads["II"] = assemble_lead(rhythm_block, "II", cal, is_rhythm=True)
    else:
        short_ii = blocks.get("II")
        if short_ii is not None and short_ii.size > 0:
            short_signal = assemble_lead(short_ii, "II", cal)
            t0, t1 = LEAD_TEMPORAL_SLOTS["II"]
            full = np.full(SAMPLES_FULL, np.nan)
            full[t0:t1] = short_signal
            leads["II"] = full
        else:
            leads["II"] = np.zeros(SAMPLES_FULL)

    # Ensure all 12 leads present
    for name in ALL_LEAD_NAMES:
        if name not in leads:
            leads[name] = np.full(SAMPLES_FULL, np.nan)

    return leads
