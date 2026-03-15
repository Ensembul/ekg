"""
Column-wise signal extraction from binary lead blocks.

Core algorithm:
  1. Clean the binary block: remove small connected components (noise dots).
  2. First pass: get rough signal via per-column median.
  3. Second pass (trace following): for each column, select the pixel cluster
     closest to the locally smoothed signal, rejecting text labels and grid
     artifacts that are far from the expected signal position.
  4. Interpolate gaps.
"""

import cv2
import numpy as np


def _mask_text_labels(block: np.ndarray) -> np.ndarray:
    """
    Mask text labels at the left edge of a lead block.

    Lead name labels (e.g. "aVR", "V1") appear in the first ~15% of columns
    and are connected to the signal trace. They show up as columns with
    very wide vertical spread compared to the signal trace (which is 2-5 px).

    Strategy:
    1. Estimate the signal's median Y from the middle of the block.
    2. In the first ~15% of columns, mask pixels that are far from
       the signal baseline (they're text, not signal).
    """
    h, w = block.shape
    result = block.copy()

    # Estimate signal Y from the middle 50% of columns (text-free region)
    mid_start = w // 4
    mid_end = 3 * w // 4
    mid_ys = []
    for col in range(mid_start, mid_end, max(1, (mid_end - mid_start) // 40)):
        rows = np.where(block[:, col] > 0)[0]
        if len(rows) > 0 and (rows[-1] - rows[0]) < h * 0.15:
            mid_ys.append(float(np.median(rows)))

    if len(mid_ys) < 5:
        return block

    signal_center = float(np.median(mid_ys))
    signal_spread = float(np.std(mid_ys)) if len(mid_ys) > 1 else 10.0
    # Allow pixels within a generous band around the signal
    tolerance = max(h * 0.15, signal_spread * 4, 30)

    # Scan the first 15% of columns for text
    scan_end = max(10, int(w * 0.15))
    for col in range(scan_end):
        rows = np.where(block[:, col] > 0)[0]
        if len(rows) == 0:
            continue

        span = rows[-1] - rows[0]
        if span < h * 0.10:
            # Narrow span — likely just signal, keep it
            continue

        # Wide span — has text pixels. Keep only pixels near signal.
        for r in rows:
            if abs(r - signal_center) > tolerance:
                result[r, col] = 0

    return result


def _clean_block(block: np.ndarray) -> np.ndarray:
    """
    Remove small connected components that are likely text labels or noise.

    Strategy: keep only the largest connected component (which is the signal
    trace) if the block has enough white pixels. For very sparse blocks,
    skip cleaning entirely to avoid removing the signal.
    """
    h, w = block.shape
    total_white = np.sum(block > 0)

    # If very few white pixels, don't risk removing them
    if total_white < 10:
        return block

    num_labels, labels, comp_stats, _ = cv2.connectedComponentsWithStats(
        block, connectivity=8
    )

    if num_labels <= 1:
        return block

    # Sort components by area (descending), skip background (label 0)
    areas = comp_stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return block

    # Keep the largest component(s) — the main signal trace
    # Also keep any component with area >= 5% of the largest
    max_area = areas.max()
    min_area = max(5, int(max_area * 0.05))

    cleaned = np.zeros_like(block)
    for i in range(1, num_labels):
        comp_area = comp_stats[i, cv2.CC_STAT_AREA]
        if comp_area >= min_area:
            cleaned[labels == i] = 255

    # If cleaning removed more than 80% of white pixels, don't clean
    if np.sum(cleaned > 0) < total_white * 0.2:
        return block

    return cleaned


def _cluster_rows(rows: np.ndarray, max_gap: int = 5) -> list[np.ndarray]:
    """Split a sorted array of row indices into clusters separated by gaps."""
    if len(rows) == 0:
        return []
    clusters = []
    current = [rows[0]]
    for i in range(1, len(rows)):
        if rows[i] - rows[i - 1] <= max_gap:
            current.append(rows[i])
        else:
            clusters.append(np.array(current))
            current = [rows[i]]
    clusters.append(np.array(current))
    return clusters


def extract_signal(block: np.ndarray) -> np.ndarray:
    """
    Extract 1D signal (Y-pixel positions) from a binary lead block.

    Two-pass approach:
      Pass 1: Rough signal via per-column median (fast but noisy at labels/grid).
      Pass 2: Trace following — for each column, find the pixel cluster closest
              to the locally smoothed signal from pass 1, picking the cluster
              center instead of the raw median of all pixels.

    block: 2D uint8 array (H, W) where 255 = signal pixel

    Returns: 1D float array of length W containing Y pixel positions.
             Higher Y = lower on image = lower voltage.
             Returns array of zeros if extraction fails (>80% empty columns).
    """
    h, w = block.shape

    # Remove text labels at left edge, then clean noise components
    masked = _mask_text_labels(block)
    cleaned = _clean_block(masked)

    # Pass 1: rough signal via per-column median
    signal_p1 = np.full(w, np.nan)
    for col in range(w):
        rows = np.where(cleaned[:, col] > 0)[0]
        if len(rows) > 0:
            signal_p1[col] = float(np.median(rows))

    nan_mask = np.isnan(signal_p1)
    if nan_mask.sum() > w * 0.80 or nan_mask.all():
        return np.zeros(w)

    # Interpolate pass 1
    idx = np.arange(w)
    valid = ~nan_mask
    signal_p1_filled = signal_p1.copy()
    signal_p1_filled[nan_mask] = np.interp(
        idx[nan_mask], idx[valid], signal_p1[valid]
    )

    # Compute local smoothed reference (moving average)
    kernel_size = max(15, w // 30)
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    padded = np.pad(signal_p1_filled, pad, mode="edge")
    local_ref = np.convolve(padded, np.ones(kernel_size) / kernel_size, mode="valid")[
        :w
    ]

    # Pass 2: trace following with cluster selection
    signal = np.full(w, np.nan)
    prev_y = None

    for col in range(w):
        rows = np.where(cleaned[:, col] > 0)[0]
        if len(rows) == 0:
            continue

        # If only a few pixels with small spread, use median directly
        spread = rows[-1] - rows[0]
        if spread < h * 0.08 or len(rows) <= 3:
            signal[col] = float(np.median(rows))
            prev_y = signal[col]
            continue

        # Multiple groups of pixels — find clusters
        clusters = _cluster_rows(rows)

        # Score each cluster: prefer the one closest to the local reference
        ref_y = local_ref[col]
        if prev_y is not None:
            # Weight: 70% local reference, 30% previous column
            target_y = 0.7 * ref_y + 0.3 * prev_y
        else:
            target_y = ref_y

        best_cluster = None
        best_dist = float("inf")
        for c in clusters:
            center = float(np.median(c))
            dist = abs(center - target_y)
            if dist < best_dist:
                best_dist = dist
                best_cluster = c

        if best_cluster is not None:
            signal[col] = float(np.median(best_cluster))
            prev_y = signal[col]

    # Check quality
    nan_mask = np.isnan(signal)
    nan_ratio = nan_mask.sum() / w

    if nan_ratio > 0.80 or nan_mask.all():
        return np.zeros(w)

    # Interpolate gaps
    if nan_mask.any():
        idx = np.arange(w)
        signal[nan_mask] = np.interp(
            idx[nan_mask], idx[~nan_mask], signal[~nan_mask]
        )

    return signal
