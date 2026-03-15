"""
Diagnostic visualizations for the ECG digitization pipeline.

Usage
-----
from debug import show_deterministic, show_tracks, show_extraction, show_variance, show_grid

show_deterministic(pil_image)      # new pipeline: per-block segmentation + extraction
show_tracks(pil_image)             # are the tracks detected correctly?
show_extraction(pil_image)         # is the signal trace following the ECG?
show_variance(pil_image)           # horizontal/vertical variance profiles
show_grid(pil_image)               # Hough grid detection (viterbi pipeline)
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.signal import find_peaks

from extract import (
    LEAD_GRID,
    LEAD_SAMPLES,
    LEAD_SLOTS,
    OUTPUT_SAMPLES,
    STANDARD_H,
    STANDARD_W,
    adaptive_otsu_binarize,
    apply_text_mask,
    auto_binarize,
    calibrate_and_cut,
    compute_calibration,
    detect_grid,
    detect_tracks,
    fill_holes,
    fragmented_extraction,
    full_extraction,
    grid_calibrate,
    highpass_filter,
    lazy_extraction,
    lowpass_filter,
    mask_text_block,
    segment_leads,
    standardize,
    trim_margins,
    twopass_extraction,
    viterbi_extract,
)


def _mask_method(mask_text_first: bool | str) -> str:
    """Normalise mask_text_first to a method string."""
    if mask_text_first is True:
        return "morphological"
    if mask_text_first is False:
        return "none"
    return str(mask_text_first)


def show_deterministic(
    pil_image: Image.Image,
    extract_method: str = "twopass",
    hp_cutoff: float | None = None,
    lp_cutoff: float | None = 40.0,
) -> plt.Figure:
    """
    Diagnostic for the deterministic pipeline.

    Shows each per-lead block with the extracted signal overlaid.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    orig_h, orig_w = img_rgb.shape[:2]
    pixels_per_mV = compute_calibration(orig_w, orig_h)

    binary = auto_binarize(img_rgb)
    blocks = segment_leads(binary)

    extract_fn = {
        "twopass": twopass_extraction,
        "lazy": lazy_extraction,
        "full": full_extraction,
        "fragmented": fragmented_extraction,
    }.get(extract_method, twopass_extraction)

    fig, axes = plt.subplots(4, 3, figsize=(20, 16), squeeze=False)

    for row_idx, row_leads in enumerate(LEAD_GRID):
        for col_idx, name in enumerate(row_leads):
            ax = axes[row_idx][col_idx]
            block = blocks[name]
            masked = mask_text_block(block)

            ax.imshow(masked, cmap="gray", aspect="auto", interpolation="nearest")

            raw_y = extract_fn(masked)
            ax.plot(np.arange(len(raw_y)), raw_y, color="red", lw=0.8, alpha=0.85)

            ax.set_title(
                f"{name}  ({block.shape[1]}×{block.shape[0]} px)",
                fontsize=8,
            )
            ax.axis("off")

    filter_parts = []
    if hp_cutoff is not None:
        filter_parts.append(f"HP {hp_cutoff} Hz")
    if lp_cutoff is not None:
        filter_parts.append(f"LP {lp_cutoff} Hz")
    filter_str = f"  |  {' / '.join(filter_parts)}" if filter_parts else ""

    fig.suptitle(
        f"Deterministic pipeline [{extract_method}]  |  "
        f"{orig_w}×{orig_h} px  |  {pixels_per_mV:.1f} px/mV{filter_str}",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def show_tracks(
    pil_image: Image.Image,
    binarize: str = "color",
    mask_text_first: bool | str = True,
) -> plt.Figure:
    """
    Shows variance-detected track boundaries overlaid on the binary image,
    and each individual track alongside it.
    """
    pil_image = apply_text_mask(pil_image, _mask_method(mask_text_first))

    binary = standardize(pil_image, binarize=binarize)
    tracks = detect_tracks(binary, n_expected=len(LEAD_GRID))
    tracks = trim_margins(binary, tracks)

    n_tracks = len(tracks)

    fig, axes = plt.subplots(
        n_tracks,
        2,
        figsize=(16, 3 * n_tracks),
        gridspec_kw={"width_ratios": [2, 1]},
    )
    if n_tracks == 1:
        axes = [axes]

    for i, track in enumerate(tracks):
        # Left: full binary image with track region highlighted
        ax_full = axes[i][0]
        ax_full.imshow(binary, cmap="gray", aspect="auto")

        # Compute approximate position of this track in the full image
        h_variance = np.var(binary.astype(float), axis=1)
        peaks, _ = find_peaks(
            h_variance, height=binary.shape[1], distance=int(binary.shape[0] / 10)
        )
        ax_full.set_title(
            f"Track {i}  |  {', '.join(LEAD_GRID[i]) if i < len(LEAD_GRID) else '?'}",
            fontsize=8,
        )
        for p in peaks:
            ax_full.axhline(p, color="red", lw=0.8, alpha=0.6)
        ax_full.axis("off")

        # Right: the extracted track
        ax_track = axes[i][1]
        ax_track.imshow(track, cmap="gray", aspect="auto")
        ax_track.set_title(
            f"Track {i}  ({track.shape[1]}×{track.shape[0]} px)", fontsize=8
        )
        ax_track.axis("off")

    fig.suptitle("Track detection diagnostic", fontsize=10)
    fig.tight_layout()
    return fig


def show_extraction(
    pil_image: Image.Image,
    hp_cutoff: float | None = None,
    lp_cutoff: float | None = 40.0,
    binarize: str = "auto",
    extract_method: str = "twopass",
    mask_text_first: bool | str = False,
) -> plt.Figure:
    """
    Overlays the extracted signal (red) on each track/block image.

    For binarize="auto", uses the deterministic pipeline (show_deterministic).
    For legacy methods, uses variance-based track detection.
    For viterbi, uses Hough grid + Viterbi DP.
    """
    if binarize == "auto":
        return show_deterministic(
            pil_image, extract_method, hp_cutoff, lp_cutoff
        )

    pil_image = apply_text_mask(pil_image, _mask_method(mask_text_first))

    if extract_method == "viterbi":
        return _show_extraction_viterbi(
            pil_image, binarize, hp_cutoff, lp_cutoff
        )

    return _show_extraction_ecgtizer(
        pil_image, binarize, extract_method, hp_cutoff, lp_cutoff
    )


def _show_extraction_ecgtizer(
    pil_image: Image.Image,
    binarize: str,
    extract_method: str,
    hp_cutoff: float | None,
    lp_cutoff: float | None,
) -> plt.Figure:
    """Extraction diagnostic for ecgtizer methods (lazy/full/fragmented/twopass)."""
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

    apply_filter = hp_cutoff is not None or lp_cutoff is not None

    n_tracks = len(tracks)
    fig, axes = plt.subplots(n_tracks, 1, figsize=(20, 4 * n_tracks), squeeze=False)

    for i, (track, row_names) in enumerate(zip(tracks, LEAD_GRID)):
        ax = axes[i][0]
        raw_signal = extract_fn(track)
        raw_signal = fill_holes(raw_signal)

        ax.imshow(track, cmap="gray", aspect="auto", interpolation="nearest")
        ax.plot(
            np.arange(len(raw_signal)), raw_signal, color="red", lw=0.9, label="raw"
        )

        if apply_filter:
            track_leads = calibrate_and_cut(raw_signal, row_names)
            for j, name in enumerate(row_names):
                lead_mv = track_leads[name]
                lead_mv = lead_mv - np.median(lead_mv)
                if hp_cutoff is not None:
                    lead_mv = highpass_filter(lead_mv, cutoff_hz=hp_cutoff)
                if lp_cutoff is not None:
                    lead_mv = lowpass_filter(lead_mv, cutoff_hz=lp_cutoff)

                pulse_len = max(1, int(len(raw_signal) * 0.027))
                lead_pixel_len = (len(raw_signal) - pulse_len) // len(row_names)
                start_px = pulse_len + j * lead_pixel_len
                end_px = start_px + lead_pixel_len

                pulse = raw_signal[:pulse_len]
                pixel_zero = float(np.max(pulse))
                pixel_one = float(np.min(pulse))
                factor = pixel_zero - pixel_one if pixel_zero != pixel_one else 1.0
                filtered_y = pixel_zero - lead_mv * factor

                filt_x = np.linspace(start_px, end_px - 1, len(filtered_y))
                ax.plot(
                    filt_x,
                    filtered_y,
                    color="blue",
                    lw=0.9,
                    alpha=0.7,
                    label="filtered" if j == 0 else None,
                )

        lead_labels = ", ".join(row_names)
        title = f"Track {i}: {lead_labels}  ({track.shape[1]} cols)"
        if apply_filter:
            parts = []
            if hp_cutoff is not None:
                parts.append(f"HP {hp_cutoff} Hz")
            if lp_cutoff is not None:
                parts.append(f"LP {lp_cutoff} Hz")
            title += f"  |  {' / '.join(parts)}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    legend_text = "red=raw" + ("  blue=filtered" if apply_filter else "")
    fig.suptitle(
        f"Extraction diagnostic [{extract_method}] — {legend_text}", fontsize=10
    )
    fig.tight_layout()
    return fig


def _show_extraction_viterbi(
    pil_image: Image.Image,
    binarize: str,
    hp_cutoff: float | None,
    lp_cutoff: float | None,
) -> plt.Figure:
    """
    Extraction diagnostic for Karbasi et al. Viterbi pipeline.

    Shows each track split into 4 lead segments with the Viterbi path (red)
    overlaid. If filters are enabled, shows the filtered mV signal (blue)
    below each track panel.
    """
    # Grid detection on original image
    grid_w, grid_h = detect_grid(pil_image)

    # Binarize
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

    apply_filter = hp_cutoff is not None or lp_cutoff is not None

    n_tracks = len(tracks)
    # 2 rows per track: top = track image + viterbi path, bottom = mV signal
    fig, axes = plt.subplots(
        n_tracks * 2, 1, figsize=(20, 3.5 * n_tracks * 2), squeeze=False
    )

    colors = ["#e6194b", "#3cb44b", "#4363d8", "#f58231"]  # per-lead segment colours

    for i, (track, row_names) in enumerate(zip(tracks, LEAD_GRID)):
        ax_img = axes[i * 2][0]
        ax_mv = axes[i * 2 + 1][0]

        n_leads = len(row_names)
        track_w = track.shape[1]
        segment_w = track_w // n_leads

        # Show the track as background
        ax_img.imshow(track, cmap="gray", aspect="auto", interpolation="nearest")

        for j, name in enumerate(row_names):
            seg_start = j * segment_w
            seg_end = (j + 1) * segment_w if j < n_leads - 1 else track_w
            segment = track[:, seg_start:seg_end]

            # Viterbi extraction on segment
            raw_y = viterbi_extract(segment)

            # Overlay Viterbi path on track image (offset by segment start)
            path_x = np.arange(len(raw_y)) + seg_start
            ax_img.plot(
                path_x, raw_y,
                color=colors[j % len(colors)], lw=1.0, alpha=0.85,
                label=name,
            )

            # Draw segment boundary
            if j > 0:
                ax_img.axvline(seg_start, color="white", lw=0.5, ls="--", alpha=0.5)

            # Grid-calibrate to mV
            lead_mv = grid_calibrate(raw_y, grid_w_std, grid_h_std, LEAD_SAMPLES)
            lead_mv = lead_mv - np.median(lead_mv)

            if hp_cutoff is not None:
                lead_mv = highpass_filter(lead_mv, cutoff_hz=hp_cutoff)
            if lp_cutoff is not None:
                lead_mv = lowpass_filter(lead_mv, cutoff_hz=lp_cutoff)

            # Plot mV signal for this lead
            t = np.linspace(j * 2.5, (j + 1) * 2.5, len(lead_mv))
            ax_mv.plot(
                t, lead_mv,
                color=colors[j % len(colors)], lw=0.8,
                label=name,
            )

        ax_img.legend(fontsize=7, loc="upper right", ncol=4)
        ax_img.set_title(
            f"Track {i}: Viterbi paths  |  grid {grid_w_std:.1f}×{grid_h_std:.1f} px/large-square",
            fontsize=9,
        )
        ax_img.axis("off")

        ax_mv.set_xlabel("Time (s)", fontsize=8)
        ax_mv.set_ylabel("mV", fontsize=8)
        ax_mv.tick_params(labelsize=7)
        ax_mv.legend(fontsize=7, loc="upper right", ncol=4)
        filter_parts = []
        if hp_cutoff is not None:
            filter_parts.append(f"HP {hp_cutoff} Hz")
        if lp_cutoff is not None:
            filter_parts.append(f"LP {lp_cutoff} Hz")
        filter_str = f"  |  {' / '.join(filter_parts)}" if filter_parts else ""
        ax_mv.set_title(
            f"Track {i}: calibrated mV signals{filter_str}",
            fontsize=9,
        )
        ax_mv.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)

    fig.suptitle("Extraction diagnostic [viterbi] — Karbasi et al.", fontsize=10)
    fig.tight_layout()
    return fig


def show_variance(
    pil_image: Image.Image,
    binarize: str = "color",
    mask_text_first: bool | str = True,
) -> plt.Figure:
    """
    Shows the horizontal and vertical variance profiles used for track detection.

    Horizontal variance peaks → track row positions.
    Vertical variance → left/right signal margins.
    """
    pil_image = apply_text_mask(pil_image, _mask_method(mask_text_first))

    binary = standardize(pil_image, binarize=binarize)

    h_variance = np.var(binary.astype(float), axis=1)
    v_variance = np.var(binary.astype(float), axis=0)

    peaks, _ = find_peaks(
        h_variance, height=binary.shape[1], distance=int(binary.shape[0] / 10)
    )
    signal_cols = np.where(v_variance > 200)[0]

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # Top-left: binary image with track lines
    axes[0][0].imshow(binary, cmap="gray", aspect="auto")
    for p in peaks:
        axes[0][0].axhline(p, color="red", lw=1, alpha=0.7)
    if len(signal_cols) > 0:
        axes[0][0].axvline(signal_cols[0], color="cyan", lw=1, alpha=0.7)
        axes[0][0].axvline(signal_cols[-1], color="cyan", lw=1, alpha=0.7)
    axes[0][0].set_title(
        "Binary image  |  red=variance peaks  cyan=signal margins", fontsize=8
    )
    axes[0][0].axis("off")

    # Top-right: horizontal variance
    axes[0][1].plot(h_variance, lw=0.6, color="steelblue")
    for p in peaks:
        axes[0][1].axvline(p, color="red", lw=0.8, alpha=0.7)
    axes[0][1].set_title(f"Horizontal variance  |  {len(peaks)} peaks", fontsize=8)
    axes[0][1].set_xlabel("Row (px)", fontsize=7)
    axes[0][1].set_ylabel("Variance", fontsize=7)
    axes[0][1].tick_params(labelsize=7)

    # Bottom-left: vertical variance
    axes[1][0].plot(v_variance, lw=0.6, color="steelblue")
    if len(signal_cols) > 0:
        axes[1][0].axvline(signal_cols[0], color="cyan", lw=0.8, alpha=0.7)
        axes[1][0].axvline(signal_cols[-1], color="cyan", lw=0.8, alpha=0.7)
    axes[1][0].set_title("Vertical variance  |  cyan=signal boundaries", fontsize=8)
    axes[1][0].set_xlabel("Column (px)", fontsize=7)
    axes[1][0].set_ylabel("Variance", fontsize=7)
    axes[1][0].tick_params(labelsize=7)

    # Bottom-right: summary
    axes[1][1].axis("off")
    summary = (
        f"Image: {binary.shape[1]}×{binary.shape[0]} px\n"
        f"Horizontal peaks: {len(peaks)}\n"
        f"Signal cols: {signal_cols[0]}–{signal_cols[-1]}"
        if len(signal_cols) > 0
        else "Signal cols: none"
    )
    axes[1][1].text(0.1, 0.5, summary, fontsize=10, va="center", family="monospace")

    fig.suptitle("Variance detection diagnostic", fontsize=10)
    fig.tight_layout()
    return fig


def show_grid(
    pil_image: Image.Image,
    mask_text_first: bool | str = True,
) -> plt.Figure:
    """
    Diagnostic for the Hough grid detection used by the Viterbi pipeline.

    Shows:
      - Top-left: original image with detected Hough lines overlaid
      - Top-right: grid intensity mask used for line detection
      - Bottom-left: adaptive Otsu binarization result
      - Bottom-right: detection summary (grid sizes, line counts)
    """
    pil_image = apply_text_mask(pil_image, _mask_method(mask_text_first))

    grid_w, grid_h = detect_grid(pil_image)

    # Reproduce internal state for visualisation
    img_np = np.array(pil_image.resize((STANDARD_W, STANDARD_H), Image.LANCZOS))
    if img_np.ndim == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    H, W = gray.shape
    scale_x = STANDARD_W / pil_image.width
    scale_y = STANDARD_H / pil_image.height
    grid_w_std = grid_w * scale_x
    grid_h_std = grid_h * scale_y

    # Grid mask
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

    # Hough lines
    lines = cv2.HoughLinesP(
        grid_mask, 1, np.pi / 180,
        threshold=50,
        minLineLength=int(min(H, W) * 0.1),
        maxLineGap=10,
    )

    # Adaptive Otsu result
    adaptive_bin = adaptive_otsu_binarize(gray)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Top-left: original image with Hough lines
    display_img = img_np.copy() if img_np.ndim == 3 else cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    n_h_lines = 0
    n_v_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 15 or angle > 165:
                cv2.line(display_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                n_h_lines += 1
            elif 75 < angle < 105:
                cv2.line(display_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                n_v_lines += 1

    axes[0][0].imshow(display_img)
    axes[0][0].set_title(
        f"Hough lines  |  red=horizontal ({n_h_lines})  blue=vertical ({n_v_lines})",
        fontsize=8,
    )
    axes[0][0].axis("off")

    # Top-right: grid intensity mask
    axes[0][1].imshow(grid_mask, cmap="gray", aspect="auto")
    axes[0][1].set_title("Grid intensity mask (input to Hough)", fontsize=8)
    axes[0][1].axis("off")

    # Bottom-left: adaptive Otsu binarization
    axes[1][0].imshow(adaptive_bin, cmap="gray", aspect="auto")
    axes[1][0].set_title("Adaptive Otsu binarization (signal isolation)", fontsize=8)
    axes[1][0].axis("off")

    # Bottom-right: summary
    axes[1][1].axis("off")
    summary = (
        f"Image: {W}×{H} px (standardised)\n"
        f"Original: {pil_image.width}×{pil_image.height} px\n"
        f"\n"
        f"Grid detection:\n"
        f"  Raw grid:  {grid_w:.1f} × {grid_h:.1f} px/large-square\n"
        f"  Scaled:    {grid_w_std:.1f} × {grid_h_std:.1f} px/large-square\n"
        f"  Implied:   {0.5/grid_h_std:.4f} mV/px | {0.2/grid_w_std:.5f} s/px\n"
        f"\n"
        f"Hough lines:\n"
        f"  Horizontal: {n_h_lines}\n"
        f"  Vertical:   {n_v_lines}\n"
        f"  Total:      {n_h_lines + n_v_lines}"
    )
    axes[1][1].text(
        0.05, 0.5, summary, fontsize=10, va="center", family="monospace",
        transform=axes[1][1].transAxes,
    )

    fig.suptitle("Grid detection diagnostic [viterbi / Karbasi et al.]", fontsize=10)
    fig.tight_layout()
    return fig
