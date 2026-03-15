"""
Training-set validation against WFDB ground truth.

Uses the actual competition scoring approach: align_signals for offset
detection, then compute SNR on the aligned signals.
"""

import os
from glob import glob
from pathlib import Path

import numpy as np
import wfdb
from scipy import stats
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from calibration import SAMPLES_SHORT
from pipeline import ECGDigitizer


# ---------------------------------------------------------------------------
# Scoring (adapted from helper_code.py's approach)
# ---------------------------------------------------------------------------


def _convert_signal(x, num_quant_levels, min_amp, max_amp, max_t):
    """Convert 1D signal to 2D binary image for cross-correlation alignment."""
    x = np.asarray(x, dtype=float).copy()
    n = x.size
    A = np.zeros((num_quant_levels, max_t), dtype=float)
    for i in range(n):
        if np.isfinite(x[i]):
            level = int(
                round((x[i] - min_amp) / (max_amp - min_amp) * (num_quant_levels - 1))
            )
            level = min(max(level, 0), num_quant_levels - 1)
            A[level, i] = 1.0
    return A


def align_signals(x_ref, x_est, num_quant_levels=100, smooth=True, sigma=0.5):
    """Align estimated signal to reference using 2D cross-correlation."""
    x_ref = np.asarray(x_ref, dtype=float)
    x_est = np.asarray(x_est, dtype=float)

    min_amp = min(np.nanmin(x_ref), np.nanmin(x_est))
    max_amp = max(np.nanmax(x_ref), np.nanmax(x_est))
    max_t = max(x_ref.size, x_est.size)

    if max_amp - min_amp < 1e-10:
        return x_est.copy(), 0, 0.0

    A_ref = _convert_signal(x_ref, num_quant_levels, min_amp, max_amp, max_t)
    A_est = _convert_signal(x_est, num_quant_levels, min_amp, max_amp, max_t)

    if smooth:
        A_ref = gaussian_filter(A_ref, sigma)
        A_est = gaussian_filter(A_est, sigma)

    A_est_flipped = A_est[::-1, ::-1]
    A_cross = fftconvolve(A_ref, A_est_flipped, mode="full")
    idx_cross = np.unravel_index(np.argmax(A_cross), A_cross.shape)

    A_auto = fftconvolve(A_ref, A_ref[::-1, ::-1], mode="full")
    idx_auto = np.unravel_index(np.argmax(A_auto), A_auto.shape)

    offset_hz = idx_auto[1] - idx_cross[1]
    offset_vt = idx_auto[0] - idx_cross[0]
    offset_vt = offset_vt / (num_quant_levels - 1) * (max_amp - min_amp)

    if offset_hz < 0:
        x_est_shifted = np.concatenate((np.full(-offset_hz, np.nan), x_est))
    else:
        x_est_shifted = np.concatenate((x_est[offset_hz:], np.full(offset_hz, np.nan)))
    x_est_shifted -= offset_vt

    return x_est_shifted, offset_hz, offset_vt


def compute_snr(x_ref, x_est, keep_nans=True):
    """Compute SNR between reference and estimated signals (competition formula)."""
    x_ref = np.asarray(x_ref, dtype=float).copy()
    x_est = np.asarray(x_est, dtype=float).copy()

    n_ref = x_ref.size
    n_est = x_est.size
    if n_est < n_ref:
        x_est = np.concatenate((x_est, np.full(n_ref - n_est, np.nan)))
    elif n_est > n_ref:
        x_ref = np.concatenate((x_ref, np.full(n_est - n_ref, np.nan)))

    idx_ref = np.isfinite(x_ref)
    idx_est = np.isfinite(x_est)

    if keep_nans:
        idx = idx_ref & idx_est
    else:
        x_est[~idx_est] = 0
        idx = idx_ref

    if idx.sum() == 0:
        return float("nan")

    x_r = x_ref[idx]
    x_e = x_est[idx]
    noise = x_r - x_e

    p_signal = np.mean(x_r**2)
    p_noise = np.mean(noise**2)

    if p_signal > 0 and p_noise > 0:
        snr = 10 * np.log10(p_signal / p_noise)
    elif p_noise == 0:
        snr = float("inf")
    else:
        snr = float("nan")

    if keep_nans:
        alpha = idx.sum() / idx_ref.sum() if idx_ref.sum() > 0 else 0
        snr *= alpha

    return float(snr) if np.isfinite(snr) else 0.0


def score_lead(extracted: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Score a single lead using competition-style metrics."""
    try:
        est_aligned, offset_hz, offset_vt = align_signals(ground_truth, extracted)
    except Exception:
        return {
            "correlation": 0.0,
            "snr_db": 0.0,
            "lag_samples": 999,
            "shape_score": 0.0,
            "amplitude_score": 0.0,
            "time_score": 0.0,
            "total_score": 0.0,
        }

    snr = compute_snr(ground_truth, est_aligned)

    # Pearson correlation on overlapping non-NaN portions
    max_n = max(ground_truth.size, est_aligned.size)
    gt_p = (
        np.concatenate((ground_truth, np.full(max_n - ground_truth.size, np.nan)))
        if ground_truth.size < max_n
        else ground_truth
    )
    est_p = (
        np.concatenate((est_aligned, np.full(max_n - est_aligned.size, np.nan)))
        if est_aligned.size < max_n
        else est_aligned
    )

    valid = np.isfinite(gt_p) & np.isfinite(est_p)
    if (
        valid.sum() > 10
        and np.std(gt_p[valid]) > 1e-10
        and np.std(est_p[valid]) > 1e-10
    ):
        r, _ = stats.pearsonr(gt_p[valid], est_p[valid])
        r = float(r) if np.isfinite(r) else 0.0
    else:
        r = 0.0

    lag = abs(int(offset_hz))

    shape_score = max(0.0, r) * 60.0
    amp_score = min(20.0, max(0.0, snr)) if snr > 0 else 0.0
    time_score = max(0.0, 20.0 - lag) if lag <= 20 else 0.0

    return {
        "correlation": r,
        "snr_db": snr,
        "lag_samples": lag,
        "shape_score": shape_score,
        "amplitude_score": amp_score,
        "time_score": time_score,
        "total_score": shape_score + amp_score + time_score,
    }


def validate_record(extracted: dict[str, np.ndarray], record_path: str) -> dict:
    """Validate extracted leads against WFDB ground truth."""
    signals, fields = wfdb.rdsamp(record_path)
    sig_names = fields["sig_name"]

    scores = {}
    for i, lead_name in enumerate(sig_names):
        gt = signals[:, i]
        ex = extracted.get(lead_name)
        if ex is None:
            ex = np.zeros(SAMPLES_SHORT)
        scores[lead_name] = score_lead(ex, gt)

    return scores


def run_validation(
    train_dir: str,
    n_samples: int = 100,
    verbose: bool = True,
) -> list[dict]:
    """Run validation on a subset of training images."""
    image_files = sorted(glob(os.path.join(train_dir, "*.png")))
    if not image_files:
        raise FileNotFoundError(f"No images in {train_dir}")

    image_files = image_files[:n_samples]
    digitizer = ECGDigitizer()
    all_scores = []

    for image_path in tqdm(image_files, desc="Validating"):
        record_name = Path(image_path).stem
        record_path = os.path.join(train_dir, record_name)

        if not os.path.exists(record_path + ".hea"):
            continue

        try:
            leads = digitizer.process_image(image_path)
            scores = validate_record(leads, record_path)
            all_scores.append(scores)

            if verbose:
                avg_total = np.mean([s["total_score"] for s in scores.values()])
                avg_corr = np.mean([s["correlation"] for s in scores.values()])
                print(f"  {record_name}: score={avg_total:.1f}, corr={avg_corr:.3f}")
        except Exception as e:
            print(f"  FAILED {record_name}: {e}")
            import traceback

            traceback.print_exc()

    if not all_scores:
        print("No records validated!")
        return []

    all_totals = [s["total_score"] for rec in all_scores for s in rec.values()]
    all_corrs = [s["correlation"] for rec in all_scores for s in rec.values()]
    all_snrs = [s["snr_db"] for rec in all_scores for s in rec.values()]
    all_lags = [s["lag_samples"] for rec in all_scores for s in rec.values()]

    print(f"\n{'=' * 60}")
    print(f"Validation Summary ({len(all_scores)} records, {len(all_totals)} leads)")
    print(f"{'=' * 60}")
    print(
        f"  Total Score:  {np.mean(all_totals):.1f} / 100  (std={np.std(all_totals):.1f})"
    )
    print(f"  Correlation:  {np.mean(all_corrs):.3f}  (std={np.std(all_corrs):.3f})")
    print(f"  SNR:          {np.mean(all_snrs):.1f} dB  (std={np.std(all_snrs):.1f})")
    print(
        f"  Lag:          {np.mean(all_lags):.1f} samples  (std={np.std(all_lags):.1f})"
    )
    print(f"{'=' * 60}")

    return all_scores
