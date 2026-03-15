# Task: ECG Digitization Agent Implementation

## Objective

Build an autonomous agent that processes ECG images and generates a submission.npz file containing digitized signals for all 12 leads across 500 test images.

**Target Performance:** 88-92 pts average score (out of 100)

---

## Agent Requirements

### Input
- Directory: `test/` containing 500 ECG images (.png/.jpg)
- Directory: `train/` containing 3000 images + ground truth (.dat, .hea files) for validation

### Output
- File: `submission.npz` containing:
  - Keys: `{record_name}_{lead_name}` (e.g., "ecg_test_0001_I")
  - Values: 1D numpy arrays (np.float16, 500 Hz sampling)
  - Total entries: 500 images × 12 leads = 6000 signals

### Success Criteria
- **Signal Shape (60 pts):** Pearson correlation > 0.85
- **Amplitude (20 pts):** SNR > 12 dB
- **Time Calibration (20 pts):** Cross-correlation lag < 15 samples
- **Completeness:** All 6000 leads present in submission

---

## Project Structure

Create the following file structure:

```
ecg_digitization/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Stage 1: Image enhancement
│   ├── grid_detection.py     # Stage 2: Grid calibration
│   ├── binarization.py       # Stage 3: Signal isolation
│   ├── segmentation.py       # Stage 4: Lead separation
│   ├── extraction.py         # Stage 5: Signal extraction
│   ├── calibration.py        # Stage 6: Amplitude/time calibration
│   ├── resampling.py         # Stage 7: 500 Hz resampling
│   └── digitizer.py          # Main pipeline orchestrator
├── validation/
│   ├── __init__.py
│   └── validate.py           # Ground truth validation
├── utils/
│   ├── __init__.py
│   └── helpers.py            # Utility functions
├── main.py                   # Entry point
├── requirements.txt
└── README.md
```

---

## Implementation Tasks

### TASK 1: Environment Setup

**File: `requirements.txt`**

Create with the following dependencies:
```
numpy>=1.21.0
opencv-python>=4.5.0
scikit-image>=0.18.0
scipy>=1.7.0
wfdb>=4.0.0
tqdm>=4.62.0
matplotlib>=3.4.0
```

**Action:** Install dependencies
```bash
pip install -r requirements.txt
```

---

### TASK 2: Implement Preprocessing Module

**File: `src/preprocessing.py`**

Implement the following functions:

#### Function 1: `load_and_assess(image_path) -> (image, degradation_level)`

```python
def load_and_assess(image_path):
    """
    Load ECG image and determine degradation level.
    
    Returns:
        image: BGR image (numpy array)
        degradation_level: str ('easy', 'medium', 'hard')
    
    Logic:
        - Load with cv2.imread()
        - Convert to grayscale
        - Calculate brightness (mean) and contrast (std)
        - Classify:
            easy: brightness 80-200, contrast > 30
            medium: brightness outside 80-200 OR contrast 20-30
            hard: contrast < 20 OR extreme brightness
    """
    pass
```

#### Function 2: `reduce_noise(image, degradation_level) -> image`

```python
def reduce_noise(image, degradation_level):
    """
    Apply noise reduction based on degradation.
    
    Args:
        image: BGR image
        degradation_level: 'easy', 'medium', or 'hard'
    
    Returns:
        denoised: BGR image
    
    Logic:
        easy: cv2.medianBlur(image, 3)
        medium: cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)
        hard: cv2.fastNlMeansDenoisingColored() + bilateralFilter
    """
    pass
```

#### Function 3: `enhance_contrast(image) -> image`

```python
def enhance_contrast(image):
    """
    Apply CLAHE for contrast enhancement.
    
    Logic:
        1. Convert BGR to LAB
        2. Apply CLAHE to L channel (clipLimit=3.0, tileGridSize=(8,8))
        3. Merge channels and convert back to BGR
    """
    pass
```

#### Function 4: `remove_text_artifacts(binary_mask) -> binary_mask`

```python
def remove_text_artifacts(binary_mask):
    """
    Remove text overlays using connected components.
    
    Logic:
        1. cv2.connectedComponentsWithStats()
        2. Filter components:
           - Keep: area >= 100, aspect_ratio >= 2.0 (signal)
           - Remove: smaller, more compact components (text)
        3. Apply morphological closing (5x3 kernel) to reconnect
    """
    pass
```

**Testing:** Verify on 5 training images (1 easy, 2 medium, 2 hard)

---

### TASK 3: Implement Grid Detection Module

**File: `src/grid_detection.py`**

#### Function 1: `detect_grid(image) -> (calibration, h_lines, v_lines)`

```python
def detect_grid(image):
    """
    Detect grid lines and extract calibration parameters.
    
    Returns:
        calibration: dict with keys:
            - 'pixels_per_mm_x': float
            - 'pixels_per_mm_y': float
            - 'pixels_per_mv': float (= 10 * pixels_per_mm_y)
            - 'pixels_per_40ms': float (= pixels_per_mm_x)
        h_lines: list of y-coordinates
        v_lines: list of x-coordinates
    
    Logic:
        1. Convert to grayscale
        2. Canny edge detection (50, 150)
        3. HoughLinesP (threshold=150, minLineLength=200)
        4. Classify lines: angle < 10° or > 170° = horizontal
                          80° < angle < 100° = vertical
        5. Calculate median spacing between consecutive lines
        6. Fallback: if detection fails, use FFT-based estimation
    """
    pass
```

#### Function 2: `remove_grid(image, calibration) -> (gray, calibration)`

```python
def remove_grid(image, calibration):
    """
    Remove grid using color filtering + morphology.
    
    Logic:
        1. Convert to HSV
        2. Create masks for pink/red/blue grids:
           - Pink: H[140-180], S[50-255], V[50-255]
           - Red: H[0-10], S[50-255], V[50-255]
           - Blue: H[90-130], S[50-255], V[50-255]
        3. Combine masks, invert to get signal mask
        4. Apply mask to image
        5. Convert to grayscale
        6. Morphological grid removal:
           - Horizontal kernel: (3*pixels_per_mm_x, 1)
           - Vertical kernel: (1, 3*pixels_per_mm_y)
           - Apply opening, subtract from image
    
    Returns:
        gray: grayscale image with grid removed
        calibration: same dict (may be updated)
    """
    pass
```

**Testing:** Verify calibration accuracy on 10 training images
- Check pixels_per_mm matches expected values (typically 15-25 pixels/mm)

---

### TASK 4: Implement Binarization Module

**File: `src/binarization.py`**

#### Function 1: `binarize_signal(gray_image, degradation_level) -> binary`

```python
def binarize_signal(gray_image, degradation_level):
    """
    Convert to binary mask.
    
    Logic:
        easy: cv2.threshold with THRESH_BINARY_INV + THRESH_OTSU
        medium/hard: cv2.adaptiveThreshold
                     (ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=15, C=5)
    """
    pass
```

#### Function 2: `refine_binary_mask(binary, degradation_level) -> binary`

```python
def refine_binary_mask(binary, degradation_level):
    """
    Refine binary mask with morphology.
    
    Logic:
        1. Opening (3x3 kernel) - remove noise
        2. If medium/hard: Closing (7x3 kernel) - connect gaps
        3. If hard: Dilation (5x3 kernel, 1 iteration)
        4. Call remove_text_artifacts()
    """
    pass
```

**Testing:** Visual inspection of binary masks on 10 training images

---

### TASK 5: Implement Segmentation Module

**File: `src/segmentation.py`**

#### Function 1: `detect_ecg_layout(binary_mask) -> (column_gaps, row_gaps)`

```python
def detect_ecg_layout(binary_mask):
    """
    Detect 4x3 lead layout.
    
    Logic:
        1. Sum along axis to get projections:
           - vertical_projection = sum(mask, axis=0)
           - horizontal_projection = sum(mask, axis=1)
        2. Find gaps (low signal regions):
           - Threshold = 10th percentile
           - Detect transitions from low to high signal
        3. Record gap center positions
        4. Fallback: if < 3 column gaps, divide width equally
           if < 2 row gaps, divide height equally
    
    Returns:
        column_gaps: [x1, x2, x3] (3 gaps for 4 columns)
        row_gaps: [y1, y2] (2 gaps for 3 rows)
    """
    pass
```

#### Function 2: `segment_leads(binary_mask, column_gaps, row_gaps) -> lead_blocks`

```python
def segment_leads(binary_mask, column_gaps, row_gaps):
    """
    Slice image into 12 lead blocks.
    
    Logic:
        1. Create boundaries:
           x_boundaries = [0] + column_gaps + [width]
           y_boundaries = [0] + row_gaps + [height]
        2. Iterate row by row, col by col (total 12 blocks)
        3. Extract blocks: mask[y1:y2, x1:x2]
        4. Store in dict with lead names
    
    Returns:
        lead_blocks: dict {
            'I': {'block': array, 'x_offset': int, 'y_offset': int},
            'II': {...},
            ...
            'V6': {...}
        }
        
    Lead order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
    Layout: row 0 = [I, II, III, aVR]
            row 1 = [aVL, aVF, V1, V2]
            row 2 = [V3, V4, V5, V6]
    """
    pass
```

**Testing:** Verify 12 blocks extracted correctly on 5 training images

---

### TASK 6: Implement Extraction Module

**File: `src/extraction.py`**

#### Function 1: `extract_signal_columnwise(lead_block) -> signal_y`

```python
def extract_signal_columnwise(lead_block):
    """
    Extract signal using column-by-column method (CORE ALGORITHM).
    
    Logic:
        1. For each column x in range(width):
           a. Get white pixels: np.where(column == 255)[0]
           b. If no pixels: append np.nan
           c. If 1 pixel: append that y-coordinate
           d. If multiple: append np.median(pixels)
        2. Interpolate NaN gaps: np.interp()
        3. Invert y-axis: signal_y = height - signal_y
    
    Returns:
        signal_y: 1D array of y-coordinates (pixels)
    """
    pass
```

#### Function 2: `extract_signal_viterbi(lead_block) -> signal_y`

```python
def extract_signal_viterbi(lead_block):
    """
    Extract signal using Viterbi dynamic programming (for hard cases).
    
    Logic:
        1. Initialize cost matrix (height x width)
        2. Set first column costs based on white pixels
        3. Forward pass (for each column x from 1 to width):
           For each y position:
               - Find minimum cost from previous column (window ±20 pixels)
               - Add smoothness penalty: (y - prev_y)^2
               - Subtract signal bonus if white pixel: -100
               - Add off-signal penalty if black: +50
        4. Backward pass: trace optimal path
        5. Invert y-axis
    
    Returns:
        signal_y: 1D array of y-coordinates (pixels)
    """
    pass
```

#### Function 3: `extract_signal_adaptive(lead_block, degradation_level) -> signal_y`

```python
def extract_signal_adaptive(lead_block, degradation_level):
    """
    Choose extraction method based on degradation.
    
    Logic:
        easy/medium: use extract_signal_columnwise()
        hard: 
            - Try columnwise first
            - Check gap_ratio = count(NaN) / length
            - If gap_ratio > 0.1: use extract_signal_viterbi()
            - Else: use columnwise result
    """
    pass
```

**Testing:** Extract signals from 20 training leads, visual inspection

---

### TASK 7: Implement Calibration Module

**File: `src/calibration.py`**

#### Function 1: `correct_baseline(signal_y) -> signal_centered`

```python
def correct_baseline(signal_y):
    """
    Center signal around zero.
    
    Logic:
        baseline = np.mean(signal_y)
        return signal_y - baseline
    """
    pass
```

#### Function 2: `calibrate_amplitude(signal_pixels, calibration) -> signal_mv`

```python
def calibrate_amplitude(signal_pixels, calibration):
    """
    Convert pixels to millivolts.
    
    Logic:
        1. Get pixels_per_mv from calibration
        2. If None or 0:
           a. Estimate: signal_range_pixels / 2.5 (expected ECG range)
           b. Update calibration
        3. Return signal_pixels / pixels_per_mv
    
    Returns:
        signal_mv: 1D array in millivolts
    """
    pass
```

#### Function 3: `align_to_grid_start(signal, lead_info, calibration) -> (signal, lead_info)`

```python
def align_to_grid_start(signal, lead_info, calibration):
    """
    Calculate temporal offset for time calibration.
    
    Logic:
        1. x_offset = lead_info['x_offset']
        2. mm_offset = x_offset / pixels_per_mm_x
        3. time_offset_s = mm_offset / 25.0  # 25 mm/s paper speed
        4. Store in lead_info['time_offset']
    
    Returns:
        signal: unchanged
        lead_info: with added 'time_offset' key
    """
    pass
```

**Testing:** Verify amplitude scaling on 10 training leads
- Compare extracted mV values to ground truth
- Should be within 20% error

---

### TASK 8: Implement Resampling Module

**File: `src/resampling.py`**

#### Function 1: `resample_to_500hz(signal, current_length_s) -> resampled`

```python
def resample_to_500hz(signal, current_length_s=2.5):
    """
    Resample to exactly 500 Hz using cubic interpolation.
    
    Logic:
        1. current_samples = len(signal)
        2. target_samples = int(current_length_s * 500)
        3. Create time arrays:
           current_time = linspace(0, current_length_s, current_samples)
           target_time = linspace(0, current_length_s, target_samples)
        4. Use scipy.interpolate.interp1d(kind='cubic')
        5. Return resampled signal
    
    Args:
        signal: 1D array (any length)
        current_length_s: duration in seconds (2.5 or 10.0)
    
    Returns:
        resampled: 1D array with exactly (current_length_s * 500) samples
    """
    pass
```

#### Function 2: `determine_lead_duration(lead_name) -> duration`

```python
def determine_lead_duration(lead_name):
    """
    Return expected duration for lead.
    
    Logic:
        If lead_name == 'II': return 10.0  # rhythm strip
        Else: return 2.5  # standard leads
    
    Note: May need adjustment based on actual dataset
    """
    pass
```

**Testing:** Verify output lengths
- 2.5s leads → 1250 samples
- 10s leads → 5000 samples

---

### TASK 9: Implement Main Digitizer Pipeline

**File: `src/digitizer.py`**

#### Class: `ECGDigitizer`

```python
class ECGDigitizer:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.calibration = None
        self.degradation_level = 'easy'
    
    def process_image(self, image_path, record_name):
        """
        Process single ECG image.
        
        Pipeline:
            1. load_and_assess()
            2. reduce_noise()
            3. enhance_contrast()
            4. detect_grid()
            5. remove_grid()
            6. binarize_signal()
            7. refine_binary_mask()
            8. detect_ecg_layout()
            9. segment_leads()
            10. For each lead:
                a. extract_signal_adaptive()
                b. correct_baseline()
                c. calibrate_amplitude()
                d. align_to_grid_start()
                e. determine_lead_duration()
                f. resample_to_500hz()
        
        Returns:
            leads_data: dict {lead_name: signal_array}
        """
        pass
    
    def process_dataset(self, image_dir, output_path='submission.npz'):
        """
        Process entire test dataset.
        
        Logic:
            1. Get all image files
            2. For each image:
               a. Try process_image()
               b. If exception: create zero-filled placeholder
            3. Generate submission using generate_submission()
        
        Returns:
            all_leads_data: dict {record_name: {lead_name: signal}}
        """
        pass


def generate_submission(all_leads_data, output_path='submission.npz'):
    """
    Generate submission.npz file.
    
    Logic:
        1. Create dict: submission_dict = {}
        2. For each record_name, leads in all_leads_data:
           For each lead_name, signal in leads:
               key = f"{record_name}_{lead_name}"
               submission_dict[key] = signal.astype(np.float16)
        3. np.savez_compressed(output_path, **submission_dict)
    """
    pass
```

**Testing:** Process 10 test images, verify npz structure

---

### TASK 10: Implement Validation Module

**File: `validation/validate.py`**

#### Function: `validate_against_ground_truth(extracted, gt_path, lead_name) -> metrics`

```python
def validate_against_ground_truth(extracted_signal, ground_truth_path, lead_name):
    """
    Validate extracted signal against ground truth.
    
    Logic:
        1. Load ground truth: wfdb.rdsamp(ground_truth_path)
        2. Find lead index in sig_name
        3. Get gt_signal from signals[:, lead_idx]
        4. Resample if lengths differ
        5. Calculate metrics:
           a. Pearson correlation → shape_score (max 60)
           b. SNR in dB → amplitude_score (max 20)
           c. Cross-correlation lag → time_score (max 20)
        6. Return dict with all scores and metrics
    
    Returns:
        metrics: dict {
            'shape_score': float (0-60),
            'amplitude_score': float (0-20),
            'time_score': float (0-20),
            'total_score': float (0-100),
            'correlation': float,
            'snr_db': float,
            'lag_samples': int
        }
    """
    pass
```

#### Function: `run_validation(train_dir) -> scores`

```python
def run_validation(train_dir):
    """
    Run validation on training set.
    
    Logic:
        1. Initialize ECGDigitizer()
        2. For each image in train_dir:
           a. Extract all leads
           b. For each lead:
              - Validate against ground truth
              - Print scores
           c. Collect all metrics
        3. Calculate average score
        4. Print summary statistics
    
    Returns:
        scores: list of metric dicts
    """
    pass
```

**Testing:** Run on 100 training images, target avg > 85 pts

---

### TASK 11: Implement Main Entry Point

**File: `main.py`**

```python
#!/usr/bin/env python3
"""
ECG Digitization Pipeline - Main Entry Point
"""

import argparse
from src.digitizer import ECGDigitizer
from validation.validate import run_validation


def main():
    parser = argparse.ArgumentParser(description='ECG Digitization Agent')
    parser.add_argument('--mode', choices=['train', 'test'], required=True,
                       help='Run validation on training set or generate test submission')
    parser.add_argument('--input_dir', required=True,
                       help='Directory containing images')
    parser.add_argument('--output', default='submission.npz',
                       help='Output file path for test mode')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("Running validation on training set...")
        scores = run_validation(args.input_dir)
        avg_score = sum(s['total_score'] for s in scores) / len(scores)
        print(f"\nFinal Average Score: {avg_score:.2f} / 100")
        
        if avg_score >= 85:
            print("✓ Target achieved! Ready for test submission.")
        else:
            print("✗ Below target. Tune parameters and retry.")
    
    else:  # test mode
        print("Processing test dataset...")
        digitizer = ECGDigitizer()
        digitizer.process_dataset(args.input_dir, args.output)
        print(f"✓ Submission saved to {args.output}")


if __name__ == '__main__':
    main()
```

**Usage:**
```bash
# Validate on training set
python main.py --mode train --input_dir train/

# Generate test submission
python main.py --mode test --input_dir test/ --output submission.npz
```

---

### TASK 12: Edge Case Handling

**File: `utils/helpers.py`**

Implement helper functions:

```python
def handle_failed_extraction(record_name, lead_name):
    """Return zero-filled signal for failed extractions"""
    return np.zeros(1250, dtype=np.float16)


def validate_signal(signal, lead_name, record_name):
    """
    Validate signal before submission.
    
    Checks:
        1. No NaN or Inf values
        2. Not all zeros
        3. Reasonable amplitude range (-5 to 5 mV)
        4. Standard deviation > 0.01
    
    Returns:
        is_valid: bool
        cleaned_signal: signal with fixes applied
    """
    pass


def estimate_calibration_fallback(image_shape):
    """
    Provide fallback calibration when detection fails.
    
    Assumes standard resolution and returns typical values:
        pixels_per_mm_x: 20.0
        pixels_per_mm_y: 20.0
    """
    pass
```

---

## Agent Execution Plan

### Phase 1: Core Implementation (Priority 1)
1. Set up environment (TASK 1)
2. Implement all modules (TASK 2-8)
3. Integrate pipeline (TASK 9)
4. Create entry point (TASK 11)

**Deliverable:** Working pipeline that processes images end-to-end

**Timeline:** 2-3 days

### Phase 2: Validation & Tuning (Priority 2)
1. Implement validation (TASK 10)
2. Run on 100 training images
3. Identify weak points (low scores)
4. Tune parameters:
   - Threshold values
   - Kernel sizes
   - Interpolation methods

**Target:** Average score > 85 pts on training set

**Timeline:** 2-3 days

### Phase 3: Edge Cases & Robustness (Priority 3)
1. Implement error handling (TASK 12)
2. Test on difficult training examples
3. Add fallback mechanisms:
   - Grid detection failure → use defaults
   - Extraction failure → zero-fill
   - Calibration failure → estimate

**Target:** 100% completion (all 6000 leads in submission)

**Timeline:** 1-2 days

### Phase 4: Final Testing & Submission (Priority 4)
1. Process all 500 test images
2. Validate submission.npz format:
   - Correct keys
   - Correct dtypes (float16)
   - Correct sampling (500 Hz)
3. Check file size (< 50 MB recommended)
4. Submit

**Timeline:** 1 day

---

## Quality Gates

### Gate 1: Core Pipeline (After Phase 1)
- [ ] All modules implemented
- [ ] No import errors
- [ ] Processes 1 image successfully
- [ ] Generates valid npz file

### Gate 2: Validation (After Phase 2)
- [ ] Average score on training set > 85 pts
- [ ] Pearson correlation > 0.85
- [ ] SNR > 12 dB
- [ ] Lag < 15 samples

### Gate 3: Robustness (After Phase 3)
- [ ] Processes all training images without crashes
- [ ] Handles grid detection failures gracefully
- [ ] Validates all signals before submission
- [ ] Zero-fills failed extractions

### Gate 4: Submission Ready (After Phase 4)
- [ ] submission.npz contains 6000 entries
- [ ] All keys formatted correctly
- [ ] All values are float16, 500 Hz
- [ ] No NaN/Inf values
- [ ] File size reasonable

---

## Performance Optimization (Optional)

### Parallel Processing

```python
from multiprocessing import Pool

def process_batch(image_paths):
    """Process multiple images in parallel"""
    with Pool(processes=4) as pool:
        results = pool.map(process_single_image, image_paths)
    return results
```

### Caching

```python
# Cache grid calibration for same resolution images
calibration_cache = {}

def get_calibration(image_shape):
    if image_shape in calibration_cache:
        return calibration_cache[image_shape]
    # ... detect grid ...
    calibration_cache[image_shape] = calibration
    return calibration
```

---

## Debugging Tools

### Visualization Helper

```python
def visualize_pipeline_stages(image_path):
    """
    Visualize each pipeline stage for debugging.
    
    Saves:
        - stage1_preprocessing.png
        - stage2_grid_removed.png
        - stage3_binary.png
        - stage4_segmented.png
        - stage5_extracted_signals.png
    """
    pass
```

### Signal Comparison

```python
def plot_comparison(extracted, ground_truth, lead_name):
    """Plot extracted vs ground truth side by side"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.plot(extracted, label='Extracted')
    plt.plot(ground_truth, label='Ground Truth', alpha=0.7)
    plt.legend()
    plt.title(f'{lead_name}')
    plt.savefig(f'comparison_{lead_name}.png')
```

---

## Expected Output

### Submission File Structure

```python
# Load and inspect submission
data = np.load('submission.npz')

# Should contain 6000 entries
assert len(data.files) == 6000

# Check format
for key in list(data.files)[:5]:
    signal = data[key]
    print(f"{key}: shape={signal.shape}, dtype={signal.dtype}")
    # Expected output:
    # ecg_test_0001_I: shape=(1250,), dtype=float16
    # ecg_test_0001_II: shape=(5000,), dtype=float16  # if 10s rhythm strip
    # ecg_test_0001_III: shape=(1250,), dtype=float16
```

### Performance Metrics (Training Set)

```
Processing 3000 images...
Average Scores:
  Signal Shape:    54.2 / 60 pts (Corr: 0.903)
  Amplitude:       17.8 / 20 pts (SNR: 16.5 dB)
  Time Calibration: 18.1 / 20 pts (Lag: 3.2 samples)
  TOTAL:           90.1 / 100 pts

✓ Target achieved!
```

---

## Agent Decision Points

### Decision 1: Degradation Classification
- **Input:** Image brightness, contrast
- **Output:** 'easy', 'medium', 'hard'
- **Impact:** Determines preprocessing intensity

### Decision 2: Extraction Method
- **Input:** Degradation level, gap ratio
- **Output:** Use columnwise or Viterbi
- **Impact:** Accuracy vs speed tradeoff

### Decision 3: Calibration Fallback
- **Input:** Grid detection success
- **Output:** Use detected or estimated calibration
- **Impact:** Amplitude and time scores

### Decision 4: Lead Duration
- **Input:** Lead name
- **Output:** 2.5s or 10.0s
- **Impact:** Resampling target length

---

## Success Metrics Summary

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Completion Rate** | 100% | 6000/6000 leads in submission |
| **Average Score** | 88-92 pts | Training set validation |
| **Shape Score** | 54+ pts | Pearson correlation > 0.85 |
| **Amplitude Score** | 17+ pts | SNR > 12 dB |
| **Time Score** | 17+ pts | Lag < 15 samples |
| **Processing Speed** | < 5 min | For 500 test images |
| **File Size** | < 50 MB | submission.npz |

---

## Final Checklist

Before submission, verify:

- [ ] All required modules implemented
- [ ] All functions have proper error handling
- [ ] Training validation score > 85 pts
- [ ] Test set processed without crashes
- [ ] submission.npz format verified
- [ ] All 6000 entries present
- [ ] All signals are float16 at 500 Hz
- [ ] No NaN or Inf values
- [ ] File size reasonable
- [ ] README.md updated with usage instructions

---

## Agent Self-Verification Commands

```bash
# Verify environment
python -c "import cv2, scipy, wfdb; print('✓ All dependencies installed')"

# Verify pipeline
python main.py --mode train --input_dir train/ | grep "Average Score"

# Verify submission format
python -c "
import numpy as np
data = np.load('submission.npz')
print(f'Entries: {len(data.files)}')
print(f'Sample: {list(data.files)[0]} → {data[list(data.files)[0]].shape}')
"

# Verify no errors
echo "✓ Ready for submission"
```

---

## Support Resources

- **Computer Vision Reference:** /mnt/user-data/outputs/ecg_digitization_approaches.md
- **Challenge Details:** /mnt/user-data/uploads/task4.pdf
- **WFDB Documentation:** https://wfdb.readthedocs.io/
- **OpenCV Documentation:** https://docs.opencv.org/

---

**Agent Directive:** Implement all tasks sequentially, validate at each gate, and ensure all success metrics are met before final submission.
