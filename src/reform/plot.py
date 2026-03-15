"""ECG signal plotting utilities."""

import matplotlib.pyplot as plt
import numpy as np

from extract import LEAD_GRID, LEAD_SAMPLES, LEAD_SLOTS, OUTPUT_SAMPLES, SAMPLE_RATE_HZ

_LEAD_DURATION_S = LEAD_SAMPLES / SAMPLE_RATE_HZ  # 2.5

_DISPLAY_ROWS = [
    ["I", "AVR", "V1", "V4"],
    ["II", "AVL", "V2", "V5"],
    ["III", "AVF", "V3", "V6"],
]


def _extract_active(signal: np.ndarray, lead_name: str) -> tuple[np.ndarray, float]:
    """
    Extract the active (non-NaN) portion of a signal and its time offset.

    If signal is 5000 samples with NaN padding, extracts the LEAD_SLOTS region.
    If signal is 1250 samples (legacy), returns as-is with offset from LEAD_SLOTS.
    """
    start, end = LEAD_SLOTS[lead_name]
    if len(signal) == OUTPUT_SAMPLES:
        mv = signal[start:end]
        t_offset = start / SAMPLE_RATE_HZ
    else:
        mv = signal
        t_offset = start / SAMPLE_RATE_HZ
    return mv, t_offset


def plot_lead(
    signal_mv: np.ndarray,
    name: str = "",
    ax=None,
) -> plt.Axes:
    """
    Plot a single lead.

    signal_mv : 1D mV array (1250 or 5000 samples)
    name      : lead name used for the title
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 2.5))

    if name and name in LEAD_SLOTS and len(signal_mv) == OUTPUT_SAMPLES:
        mv, t_offset = _extract_active(signal_mv, name)
    else:
        mv = signal_mv
        t_offset = 0.0

    t = np.arange(len(mv)) / SAMPLE_RATE_HZ + t_offset
    finite = np.isfinite(mv)
    mv_clean = np.where(finite, mv, 0.0)

    _apply_ecg_grid(ax, t[0], t[-1], _y_range(mv_clean[finite] if finite.any() else mv_clean))
    ax.plot(t[finite], mv_clean[finite], color="black", lw=0.9, zorder=3)
    ax.axhline(0, color="#cc4444", lw=0.5, linestyle=":", zorder=2)
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("mV", fontsize=8)
    if name:
        ax.set_title(name, fontsize=9)
    ax.tick_params(labelsize=7)
    return ax


def plot_ecg(
    leads: dict[str, np.ndarray],
    title: str = "",
    save: str | None = None,
    figsize: tuple = (20, 12),
) -> plt.Figure:
    """
    Plot all leads in standard 4x3 ECG layout.

    Each row spans 0-10 s, with each lead placed at its correct time offset.

    leads  : output of digitize() — {lead_name: mV array (1250 or 5000 samples)}
    title  : optional figure suptitle
    save   : file path to save (e.g. "ecg.png"), or None to display inline
    figsize: (width, height) in inches
    """
    fig, axes = plt.subplots(
        len(_DISPLAY_ROWS),
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.1},
    )
    if len(_DISPLAY_ROWS) == 1:
        axes = [axes]

    for ax, row_leads in zip(axes, _DISPLAY_ROWS):
        segments = []
        for j, name in enumerate(row_leads):
            if name not in leads:
                continue
            mv_full = leads[name]
            mv, t_offset = _extract_active(mv_full, name)
            finite = np.isfinite(mv)
            if not finite.any():
                continue
            t = np.arange(len(mv)) / SAMPLE_RATE_HZ + t_offset
            segments.append((name, t[finite], mv[finite]))

        all_mv = (
            np.concatenate([mv for _, _, mv in segments])
            if segments
            else np.array([0.0])
        )
        _apply_ecg_grid(ax, 0.0, 10.0, _y_range(all_mv))

        for name, t, mv in segments:
            ax.plot(t, mv, color="black", lw=0.8, zorder=3)
            ax.text(
                t[0] + 0.05,
                1.0,
                name,
                transform=ax.get_xaxis_transform(),
                fontsize=7,
                color="#333333",
                va="bottom",
                ha="left",
            )

        ax.axhline(0, color="#cc4444", lw=0.4, linestyle=":", zorder=2)
        ax.set_ylabel("mV", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Time (s)", fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _y_range(data_mv: np.ndarray, pad: float = 0.15) -> tuple[float, float]:
    """Returns (y_min, y_max) snapped to nearest 0.1 mV with padding."""
    lo = np.nanmin(data_mv) - pad
    hi = np.nanmax(data_mv) + pad
    lo = min(lo, -0.5)
    hi = max(hi, 0.5)
    return (np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10)


def _apply_ecg_grid(
    ax,
    x_start: float,
    x_end: float,
    y_range: tuple[float, float],
) -> None:
    """
    Applies standard ECG paper grid styling to an axes.

    Minor grid: 1 mm squares → 0.04 s × 0.1 mV
    Major grid: 5 mm squares → 0.2 s  × 0.5 mV
    """
    y_lo, y_hi = y_range

    ax.set_facecolor("#fff8f8")

    ax.set_xticks(np.arange(x_start, x_end + 0.001, 0.04), minor=True)
    ax.set_xticks(np.arange(x_start, x_end + 0.001, 0.20))
    ax.set_yticks(np.arange(y_lo, y_hi + 0.001, 0.10), minor=True)
    ax.set_yticks(np.arange(y_lo, y_hi + 0.001, 0.50))

    ax.xaxis.grid(True, which="minor", color="#ffcccc", linewidth=0.3, zorder=1)
    ax.xaxis.grid(True, which="major", color="#ff8888", linewidth=0.7, zorder=1)
    ax.yaxis.grid(True, which="minor", color="#ffcccc", linewidth=0.3, zorder=1)
    ax.yaxis.grid(True, which="major", color="#ff8888", linewidth=0.7, zorder=1)

    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(which="minor", length=0)
