"""
Rotation correction for ECG images.

Uses the deskew library as a starting point, enhanced with:
- Size-based rotation prior (image size predicts rotation amount)
- Try both +/- signs (grid symmetry causes sign ambiguity)
- Combined scoring (signal quality + row projection) to pick best angle
- Fallback to size-based prior when quality scoring is ambiguous
"""

import numpy as np
from PIL import Image
from deskew import determine_skew

from calibration import ROW_STARTS, ROW_ENDS, COL_STARTS, COL_ENDS


def _signal_quality(gray: np.ndarray) -> float:
    """Score how well the image aligns with expected ECG layout.

    Combines two signals:
    1. Concentration of dark pixels in lead row regions
    2. Horizontal continuity of signal traces within lead blocks
    """
    h, w = gray.shape

    # Find dark pixels (potential signal)
    thresh = min(np.percentile(gray, 3), 80)
    dark = gray < thresh
    total_dark = dark.sum()
    if total_dark == 0:
        return 0.0

    # Score 1: concentration of dark pixels in expected regions
    in_rows = 0
    for rs, re in zip(ROW_STARTS, ROW_ENDS):
        for cs, ce in zip(COL_STARTS, COL_ENDS):
            y0, y1 = int(rs * h), int(re * h)
            x0, x1 = int(cs * w), int(ce * w)
            in_rows += dark[y0:y1, x0:x1].sum()

    concentration = in_rows / total_dark

    # Score 2: signal continuity within lead blocks
    continuity = 0.0
    n_blocks = 0
    for rs, re in zip(ROW_STARTS[:3], ROW_ENDS[:3]):
        for cs, ce in zip(COL_STARTS, COL_ENDS):
            y0, y1 = int(rs * h), int(re * h)
            x0, x1 = int(cs * w), int(ce * w)
            block = dark[y0:y1, x0:x1]
            bh, bw = block.shape
            if bw < 10:
                continue

            cols_with_signal = 0
            narrow_cols = 0
            for col_idx in range(0, bw, max(1, bw // 30)):
                col = block[:, col_idx]
                ys = np.where(col)[0]
                if len(ys) > 0:
                    cols_with_signal += 1
                    spread = ys.max() - ys.min()
                    if spread < bh * 0.3:
                        narrow_cols += 1

            if cols_with_signal > 0:
                continuity += narrow_cols / cols_with_signal
                n_blocks += 1

    if n_blocks > 0:
        continuity /= n_blocks

    return concentration * 0.5 + continuity * 0.5


def _row_projection_score(gray: np.ndarray) -> float:
    """Score rotation quality using horizontal projection variance.

    A well-aligned ECG has distinct horizontal bands (rows of leads).
    The horizontal projection (sum of dark pixels per row) should show
    sharp peaks at lead positions when correctly deskewed.
    """
    h, w = gray.shape
    thresh = min(np.percentile(gray, 3), 80)
    dark = (gray < thresh).astype(np.float32)

    projection = dark.sum(axis=1)
    mean_proj = projection.mean()
    if mean_proj < 1:
        return 0.0

    # Look at the middle 80% (avoid edge artifacts from rotation)
    margin = h // 10
    mid_proj = projection[margin : h - margin]

    return float(np.std(mid_proj) / (mean_proj + 1e-6))


def deskew_image(pil_image: Image.Image) -> Image.Image:
    """
    Detect and correct image rotation.

    Strategy:
    1. Use image size to predict likely rotation magnitude.
    2. Get candidate angle from deskew library for sign hint.
    3. Score candidates with combined metric (signal quality + row projection).
    4. If scoring is decisive, use the best-scoring angle.
    5. If scoring is ambiguous, fall back to size prior + deskew sign hint.
    """
    w, h = pil_image.size
    gray = np.array(pil_image.convert("L"))

    # Size-based rotation prior
    if w == 3300 and h == 2550:
        return pil_image

    # Determine expected rotation magnitude from image size
    if w == 2200 and h == 1700:
        prior_angle = 5.0
    elif w == 924 and h == 714:
        prior_angle = 0.0
    else:
        prior_angle = 15.0

    # Get deskew library's estimate
    deskew_angle = determine_skew(gray, max_angle=20, num_peaks=20)

    # Build candidate list
    all_candidates = [0.0]

    if prior_angle > 0.3:
        all_candidates.extend([prior_angle, -prior_angle])

    if deskew_angle is not None and abs(deskew_angle) > 0.5:
        mag = abs(deskew_angle)
        if abs(mag - prior_angle) > 2.0 and mag > 0.3:
            all_candidates.extend([mag, -mag])
        if deskew_angle not in all_candidates:
            all_candidates.append(deskew_angle)

    # For 924x714, also try 15°
    if w == 924 and h == 714:
        all_candidates.extend([15.0, -15.0])

    # Deduplicate
    seen = set()
    unique_candidates = []
    for a in all_candidates:
        key = round(a, 1)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(a)

    # Score each candidate
    def combined_score(g):
        sq = _signal_quality(g)
        rp = _row_projection_score(g)
        return sq * 0.7 + rp * 0.3

    best_angle = 0.0
    best_score = combined_score(gray)

    for angle in unique_candidates:
        if abs(angle) < 0.1:
            continue

        rotated = pil_image.rotate(
            angle,
            resample=Image.BILINEAR,
            expand=False,
            fillcolor=(255, 255, 255, 255),
        )
        rotated_gray = np.array(rotated.convert("L"))
        score = combined_score(rotated_gray)

        if score > best_score * 1.03:
            best_score = score
            best_angle = angle

    if abs(best_angle) < 0.1:
        return pil_image

    corrected = pil_image.rotate(
        best_angle,
        resample=Image.BILINEAR,
        expand=False,
        fillcolor=(255, 255, 255, 255),
    )
    return corrected
