# ECG Digitization Pipeline — Technical Report

## Problem

Recover 12-lead digital ECG signals (500 Hz) from photographs/scans of paper ECG printouts. Images range from clean 300 DPI scans to crumpled, rotated, low-res photocopies.

## Pipeline

### 1. Calibration

Image dimensions determine DPI and grid scale. The ECG layout is standardized (US Letter, 25 mm/s, 10 mm/mV), so all spatial conversions are derived directly from image width — no grid-line detection needed.

### 2. Rotation Correction

Expected rotation is estimated from image size. A candidate angle from the `deskew` library is scored (70% signal quality, 30% row projection variance) and accepted only if it improves the score by at least 3%.

### 3. Binarization

- **Color images**: HSV filtering separates grid (saturated + bright) from signal (dark, non-grid). Morphological open/close cleans noise.
- **B&W images**: Adaptive Gaussian thresholding + horizontal erosion/dilation to break vertical grid lines while preserving the trace.

### 4. Signal Extraction

The binary image is sliced into 13 blocks (12 leads + rhythm strip) using fixed layout fractions, then a two-pass method extracts each trace:

1. **Pass 1**: Median Y of signal pixels per column (rough estimate).
2. **Pass 2**: Cluster signal pixels per column, select the cluster closest to a weighted reference (70% smoothed Pass-1 + 30% previous column) to reject text/grid residuals.

### 5. Assembly

- Pixel positions are converted to millivolts via calibration constants.
- Cubic interpolation resamples to 1250 samples (short leads) or 5000 samples (Lead II rhythm strip).
- 40 Hz low-pass Butterworth filter removes quantization noise.
- Short leads are placed at their correct temporal offset; unobserved regions are NaN-filled.

## Scoring

| Component | Metric | Max |
|-----------|--------|-----|
| Shape | Pearson correlation | 60 |
| Amplitude | SNR (dB) | 20 |
| Time | Cross-correlation lag | 20 |

## Key Decisions

- **Classical CV only** — no deep learning models; lightweight and reproducible.
- **NaN padding** over zero-fill for unobserved regions (~6 dB SNR improvement).
- **No high-pass filter** — distorts short 2.5 s leads; median subtraction used instead.
