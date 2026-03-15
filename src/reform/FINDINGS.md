# ECG Digitization Pipeline — Key Findings

Findings from the `the_last_hope` pipeline implementation across three iterative sessions.

---

## Calibration

Calibration is fully deterministic from image dimensions — no grid detection needed.

- **DPI**: `width / 11` (11-inch standard ECG paper width)
- **Spatial**: `pixels_per_mm = DPI / 25.4`
- **Amplitude**: `pixels_per_mV = pixels_per_mm × 10` (standard 10 mm/mV gain)
- **Layout fractions are fixed**:
  - Columns: `[0.054, 0.277, 0.501, 0.725]` start, `[0.277, 0.501, 0.725, 0.946]` end
  - Rows: `[0.334, 0.500, 0.667, 0.834]` start, `[0.500, 0.667, 0.834, 1.000]` end
  - Rhythm strip: full width, row index 3

---

## Image Categories

Augmentation tier is perfectly predictable from image size:

| Size | DPI | Rotation | Type | Train / Test (%) |
|------|-----|----------|------|------------------|
| 3300×2550 | 300 | 0° | Clean | 1600 / 110 (22%) |
| 2200×1700 | 200 | 5° | Temp-shifted, noisy | 1000 / 195 (39%) |
| 924×714 | 84 | 0° | Black & white | 92 / 123 (25%) |
| Other | 50–200 | 15° | Wrinkles, handwriting, heavy noise | 308 / 72 (14%) |

---

## Binarization

### Colored images
- Primary signal detector: **grayscale darkness** (`gray < dark_thresh`).
- Grid exclusion: `S > 30 AND V > 100` marks colored bright pixels as grid.
- `dark_thresh = min(max(percentile(gray, 3) × 1.5, 30), 100)`.
- Works across normal and temperature-shifted images because the signal trace is always the darkest feature in grayscale.

### BW images (84 DPI)
- **Adaptive thresholding** (`blockSize=91, C=12`, Gaussian) captures the signal as the locally darkest feature.
- Followed by horizontal erosion/dilation (`3×1` kernel) to break thin vertical grid lines.
- **Critical**: skip the noise cleanup (morphological open/close) for BW images — it removes thin signal pixels at low DPI.
- Original morphological grid removal (horizontal/vertical opening with `grid_px×4` kernels) was destroying the signal because at 84 DPI the trace is continuous and wider than the kernel.

### Temperature shift impact on binarization
- Warm shift (temp < 5000): signal trace has moderate saturation (S ≈ 86–100), low V (< 100). Correctly passes the grid filter but fewer pixels captured (3.3% vs 8% on clean).
- Cool shift (temp > 15000): grid lines drop to V = 66–100, falling below the `V > 100` grid filter. Grid pixels leak into the signal mask. This remains an unsolved problem.
- The `_is_temp_shifted` detection function (using V dark fraction vs grayscale dark fraction) was unreliable — only caught 4/30 temp-shifted images. Removed in favour of the unified grayscale approach.

---

## Signal Extraction

### Text label masking (+7.5 points on clean images)
- Lead name labels ("aVR", "V1", etc.) are connected to the signal trace as a single connected component, so connected-component filtering cannot remove them.
- Labels occupy the first ~5–16% of each block's columns and have much wider vertical spread than the signal trace.
- Solution: estimate signal baseline Y from the middle 50% of columns, then mask pixels far from baseline in the first 15% of columns.
- Impact: aVR 59→79, aVL 59→74, aVF 56→74, V1 67→80.

### Two-pass extraction
- **Pass 1**: rough signal via per-column median of white pixels.
- **Pass 2**: trace-following — for each column, cluster the white pixels, select the cluster closest to the locally smoothed Pass 1 reference (70% local ref, 30% previous column).
- Handles multi-cluster columns (signal + residual grid/text) well.

### Baseline estimation
- **Global median** baseline outperforms running median and highpass filtering for short leads (2.5s).
- Running median (500-sample window) distorts signal shape on short segments.
- Highpass at 0.5 Hz helps some leads marginally but hurts others — net negative.

---

## Deskew / Rotation Correction

- Quality scoring function: `signal_quality × 0.7 + row_projection_score × 0.3`.
- Candidate angles: `[0, ±prior_angle, ±deskew_library_angle]`.
- Accept a rotation only if it beats the current best by **1.03×** (3% improvement threshold).
- **Do not force rotation based on expected angle**: for ~50% of rot5 images, angle=0 gives a better competition score than ±5°. The fallback logic that forced rotation was actively harmful (e.g., 42.0 vs 4.4 on individual images).

---

## Scoring Formula

```
total = max(0, pearson_r) × 60 + min(20, max(0, snr_db)) + max(0, 20 - abs(lag))
```

### Component breakdown (clean images, final pipeline)
| Component | Score | Max | Percentage |
|-----------|-------|-----|------------|
| Shape (correlation) | 55.4 | 60 | 92% |
| Amplitude (SNR) | 3.5 | 20 | 17.5% |
| Time (lag) | 18.3 | 20 | 91.5% |

### Alpha penalty — the structural bottleneck
- GT has **5000 finite samples for ALL 12 leads** (full 10s waveform).
- Short leads only show 2.5s on the image → 1250 extracted samples + 3750 NaN.
- `alpha = both_finite / ref_finite = 1250 / 5000 = 0.25`.
- `snr_effective = snr_raw × alpha` → even perfect extraction caps at ~5–7 dB for short leads.
- This is why Lead II (rhythm strip, 5000 samples, alpha=1.0) consistently scores highest.

### NaN fill is optimal
| Strategy | SNR (perfect extraction) |
|----------|--------------------------|
| NaN fill | 6.80 dB |
| Zero fill | 1.10 dB |
| Mean fill | 0.70 dB |
| Tiled 4× | −1.78 dB |

### Filtering
- **40 Hz lowpass filter helps**: GT has < 0.5% power above 40 Hz. The filter removes pixel quantization noise.
- No highpass filter — global median baseline subtraction is sufficient.

---

## Per-Lead Score Patterns (Clean Images)

| Lead | Avg Score | Notes |
|------|-----------|-------|
| II | 88.3 | Rhythm strip, no alpha penalty |
| I | 79.1 | |
| III | 78.7 | |
| aVR | 78.8 | Was 59.2 before text masking |
| aVL | 74.4 | Was 58.9 before text masking |
| aVF | 74.1 | Was 55.5 before text masking |
| V1 | 80.4 | Was 67.4 before text masking |
| V2 | 76.8 | |
| V3 | 77.0 | |
| V4 | 72.2 | Rightmost column, often clipped |
| V5 | 73.0 | |
| V6 | 73.7 | |

---

## Score Progression

| Stage | Clean | Rot5 | BW | Rot15 | Weighted est. |
|-------|-------|------|----|-------|---------------|
| Initial | 71.3 | 22.8 | 2.7 | 3.0 | ~25 |
| +BW adaptive threshold | 71.3 | 22.8 | 28.9 | 3.0 | ~30 |
| +Unified color binarization | 69.8 | 35.5 | 22.3 | 8.7 | ~36 |
| +Text label masking | 77.2 | 45.8 | 23.8 | 11.9 | ~45 |

---

## Unsolved Problems

1. **Extreme temperature shifts** (temp < 4000 or > 16000): grid lines and signal have similar grayscale darkness AND similar V values (V = 66–100), making HSV separation unreliable. The `V > 100` grid filter misses these grid pixels.

2. **Tall R-wave clipping**: peaks > 3.5 mV extend beyond block boundaries (e.g., 500 px peak vs 424 px block height). Adjacent lead rows are contiguous with no gap, so expanding boundaries risks capturing the wrong lead's signal.

3. **Rot15 images** (14% of test set): wrinkles, handwriting overlay, noise up to 40, and 15° rotation make extraction fundamentally difficult. Average score 11.9.

4. **Alpha penalty is structural**: short leads can never score above ~7 dB SNR regardless of extraction quality, because the GT expects 5000 samples but only 1250 are visible on the image.

5. **Lead II on rot5**: rhythm strip scores only 32.6 avg on rot5 images (vs 88.3 on clean), dragging down the overall score significantly. Root cause appears to be binarization quality degradation at 200 DPI with temperature shift + noise.
