# ECG Paper Digitization: A Comprehensive Research Summary

This document provides a comprehensive overview of recent approaches to ECG (electrocardiogram) paper digitization, focusing on five key areas: preprocessing methods, model architectures, converting detections to 1D signals, detecting units from gridlines, and aligning detections with units.

## Table of Contents
1. [Overview and Context](#overview-and-context)
2. [Preprocessing Methods](#preprocessing-methods)
3. [Model Architectures](#model-architectures)
4. [Converting Detection to 1D Signal](#converting-detection-to-1d-signal)
5. [Detecting Units from Gridlines](#detecting-units-from-gridlines)
6. [Aligning Detections with Units](#aligning-detections-with-units)
7. [Key Datasets and Resources](#key-datasets-and-resources)
8. [Open Source Tools and Repositories](#open-source-tools-and-repositories)
9. [References and Links](#references-and-links)

---

## Overview and Context

ECG digitization is the process of converting paper-based or scanned ECG images back into time-series signals. This is critical for leveraging decades of legacy clinical data in modern deep learning applications. Despite recent advances in digital ECG devices, physical paper ECGs remain common, especially in the Global South, with billions of paper ECGs archived worldwide.

### Recent Developments (2024-2026)

The field has seen significant advancement, particularly driven by the **George B. Moody PhysioNet Challenge 2024**, which focused specifically on digitizing and classifying ECG images. This challenge has catalyzed research and produced several state-of-the-art approaches.

**Key Papers and Timeline:**
- **2024**: PhysioNet Challenge 2024 launched, ECG-Image-Kit released
- **Oct 2024**: SignalSavants winning approach (Hough Transform + U-Net)
- **Dec 2024**: ECGtizer - fully automated pipeline with signal recovery
- **2025**: Multiple open-source frameworks released
- **Feb 2026**: PTB-XL-Image-17K dataset published

---

## Preprocessing Methods

Preprocessing is crucial for handling the wide variety of ECG image qualities, artifacts, and distortions encountered in real-world paper ECGs.

### 1. Image Rotation and Perspective Correction

**Hough Transform for Grid Alignment:**
- **Primary Method**: The Hough Transform is widely used to detect the grid lines in ECG paper, which enables precise rotation correction
- **Reference**: Krones et al. (2024) - "Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts"
- **Implementation**: 
  - Detects horizontal and vertical grid lines
  - Calculates rotation angle from line orientations
  - Corrects perspective distortions up to ~45° rotation
  - **SNR Achievement**: 17.02 dB average on cross-validation

**Perspective Detection Algorithms:**
- Open-ECG-Digitizer uses autocorrelation-based template matching
- Handles scanned papers and mobile phone photos
- Robust to perspective distortions, wrinkles, and stains
- Achieves mean SNR of **19.65 dB** on scanned papers with artifacts

### 2. Grid Detection and Removal

**Adaptive Grid Detection:**
- **ECG-Image-Kit** provides configurable grid detection
- Supports multiple grid colors (red, green, black, blue)
- Detects grid periodicity using autocorrelation
- Removes or masks grid lines before signal extraction

**Grid Parameters Typically Detected:**
- Horizontal spacing: 1mm (minor), 5mm (major) at 25 mm/s or 50 mm/s
- Vertical spacing: 1mm = 0.1mV (standard calibration)
- Grid color and intensity variations

### 3. Noise and Artifact Handling

**Common Preprocessing Pipeline:**
1. Grayscale conversion
2. Contrast enhancement
3. Grid detection using autocorrelation or Hough transform
4. Noise removal (wrinkles, stains, creases)
5. Region of interest (ROI) extraction

**Data Augmentation for Robustness:**
- Synthetic generation of artifacts (wrinkles, creases, stains)
- Random rotations (up to 5°)
- Random black/white level adjustments
- Text artifact overlays
- Perspective transforms

### 4. Multi-Scale Processing

**Approaches:**
- **ECGtizer** uses three pixel-based signal extraction algorithms
- Adaptive binarization for different image qualities
- Handling both clean scans and mobile phone photographs

---

## Model Architectures

Deep learning architectures for ECG digitization have evolved significantly, with a clear trend toward segmentation-based approaches.

### 1. Segmentation-Based Approaches

**U-Net Architecture (Dominant Approach):**

The U-Net architecture has emerged as the most effective for ECG signal segmentation:

**Key Implementations:**
- **nnU-Net** (Krones et al., 2024): Self-configuring U-Net variant
  - Automatically adapts to dataset characteristics
  - Achieves IoU of 0.87 for segmentation
  - Winner of PhysioNet Challenge 2024
  
- **Standard U-Net** (Karbasi et al., 2025):
  - Two-stage pipeline: segmentation → signal extraction
  - Custom data augmentations for overlapping signals
  - Handles signal overlaps (common in paper ECGs)
  - MSE: 0.0010 (non-overlapping), 0.0029 (overlapping)

**U-Net Modifications for ECG:**
- Multi-label segmentation (signal vs. grid vs. background)
- Custom loss functions for imbalanced classes
- Data augmentation specific to ECG artifacts

### 2. Detection-Based Approaches

**YOLO for Lead Detection:**
- Used for detecting individual lead regions
- Bounding box annotations in YOLO format
- Enables cropping and individual lead processing
- **PTB-XL-Image-17K** provides YOLO-format annotations

**Two-Stage Detection:**
1. Lead region detection (YOLO or similar)
2. Signal extraction within each region

### 3. End-to-End Deep Learning

**Multi-Task Architectures:**
- Some approaches combine digitization with classification
- Shared encoders for feature extraction
- Separate decoders for signal reconstruction and disease classification

**Transformer-Based Approaches:**
- Emerging use of Vision Transformers (ViT)
- Better handling of global context
- Still less common than CNN-based approaches

### 4. Hybrid Approaches

**Hough Transform + Deep Learning:**
- **SignalSavants (2024 Winner)**:
  1. Hough transform for rotation correction
  2. U-Net for segmentation
  3. Mask vectorization for signal reconstruction
  
- **Vindigitizer (2024)**:
  1. YOLO for lead detection
  2. Hough transform for grid detection
  3. Viterbi's algorithm for signal extraction

---

## Converting Detection to 1D Signal

The conversion from segmented masks or detected pixels to 1D time-series signals involves several key steps.

### 1. Pixel-to-Signal Conversion Methods

**Vertical Scanning (Most Common):**
- Scan each column of the segmented region
- Identify the signal pixel(s) in each column
- Convert pixel coordinates to voltage values based on grid scale
- **Time Resolution**: Typically 500 Hz sampling rate

**Centerline Extraction:**
- Find the center of the signal trace in each column
- Handle overlapping signals using priority rules
- Interpolate missing points

**Vectorization Approaches:**
- **Mask Vectorization** (Krones et al., 2024):
  - Convert binary segmentation masks to polylines
  - Smoothing and resampling to uniform time steps
  - Handles multi-pixel thick traces

### 2. Handling Overlapping Signals

**Challenge:**
- Paper ECGs often have overlapping signals between leads
- Traditional vertical scanning fails with overlaps

**Solutions:**
- **Karbasi et al. (2025)**: U-Net segmentation isolates primary trace
  - Trained on overlapping signal examples
  - Custom augmentations to simulate overlaps
  - Significant improvement: rho = 0.9641 vs. baseline 0.8676

- **ECGtizer**: Fragmented extraction method
  - Extracts non-overlapping segments
  - Deep learning-based completion for missing parts
  - Three extraction algorithms: full, lazy, fragmented

### 3. Signal Reconstruction

**Deep Learning-Based Reconstruction:**
- **ECGtizer** uses deep learning to recover lost signals
- Completes partial leads (2.5s or 5s) to full 10s recordings
- U-Net-based completion model
- Replaces extracted leads with completed versions

**Traditional Interpolation:**
- Linear interpolation for gaps
- Spline interpolation for smooth transitions
- Limitations with large gaps or complex morphologies

### 4. Temporal Alignment

**Cross-Correlation Methods:**
- Align leads using R-peak detection
- Ensure temporal consistency across 12 leads
- Handle variable paper speeds (25 mm/s vs. 50 mm/s)

---

## Detecting Units from Gridlines

Accurate calibration is essential for converting pixels to physiological units (mV and seconds).

### 1. Grid Detection Methods

**Autocorrelation-Based Approach:**
- Detect periodicity in image intensity
- Find dominant frequencies corresponding to grid spacing
- Robust to grid color variations

**Template Matching:**
- Predefined grid patterns
- Cross-correlation to find best match
- Used in Open-ECG-Digitizer

**Hough Transform:**
- Detect straight lines
- Filter by expected grid line spacing
- Calculate pixel-to-mm conversion factor

### 2. Calibration Parameters

**Standard ECG Calibration:**
- **Time scale**: 25 mm/s or 50 mm/s (configurable)
- **Voltage scale**: 10 mm/mV (standard), 5 mm/mV (alternate)
- **Grid spacing**: 
  - Minor: 1mm × 1mm
  - Major: 5mm × 5mm

**Calibration Pulse Detection:**
- Some ECGs include calibration pulses
- 1mV standard square wave
- Automated detection enables self-calibration
- 60% of images in Open-ECG-Digitizer have calibration pulses

### 3. Scale Estimation

**Pixel Size Estimation:**
- Count pixels between detected grid lines
- Calculate pixels-per-mm ratio
- Handle different image resolutions (DPI)
- Common: 150-500 DPI for scans

**Multi-Scale Validation:**
- Verify consistency across horizontal and vertical
- Cross-check with expected grid dimensions
- Flag uncertain calibrations

### 4. Grid Removal vs. Preservation

**Two Strategies:**
1. **Grid Removal**: Remove grid before processing (cleaner signal)
2. **Grid Preservation**: Use grid as reference during extraction

**Trade-offs:**
- Grid removal risks losing calibration reference
- Grid preservation requires robust grid detection
- Most modern approaches use grid for calibration then remove it

---

## Aligning Detections with Units

The final step is ensuring the extracted signal values accurately represent physiological measurements.

### 1. Coordinate System Transformation

**Image to Physical Coordinates:**
- Origin placement (typically top-left)
- Y-axis inversion (image coordinates vs. signal coordinates)
- Scale application:
  - X-axis: pixels → seconds (using mm/s)
  - Y-axis: pixels → millivolts (using mm/mV)

**Coordinate Systems:**
```
Image Coordinates:        Signal Coordinates:
(0,0) ───→ X                Voltage
  │                           ↑
  ↓                           │
  Y                         Time →
```

### 2. Lead Layout Identification

**Template Matching:**
- Match detected lead regions to known layouts
- Common layouts: 3×4 grid, 6×2 grid, rhythm strip
- **Open-ECG-Digitizer** uses predefined templates
- Configurable via YAML files

**Lead Label Recognition:**
- OCR for lead names (I, II, III, aVR, aVL, aVF, V1-V6)
- Pattern matching for standard labels
- YOLO-based detection in recent approaches

### 3. Temporal Synchronization

**R-Peak Alignment:**
- Detect R-peaks in each lead
- Align leads using R-peak timing
- Handle multi-page ECGs

**Sampling Rate Standardization:**
- Output at standard sampling rates (500 Hz, 1000 Hz)
- Resample extracted signals uniformly
- Handle variable input resolutions

### 4. Quality Validation

**Signal Quality Metrics:**
- Signal-to-Noise Ratio (SNR) - primary evaluation metric
- Mean Squared Error (MSE) vs. ground truth
- Pearson Correlation Coefficient
- Clinical feature preservation (QRS width, QT interval, etc.)

**Clinical Validation:**
- Verify extracted features match expected ranges
- Check for physiologically impossible values
- Cross-validation across multiple datasets

---

## Key Datasets and Resources

### 1. PTB-XL Dataset
- **Size**: 21,799 12-lead ECG recordings
- **Content**: Signal data with diagnostic labels
- **Usage**: Foundation for generating synthetic ECG images
- **Link**: https://physionet.org/content/ptb-xl/1.0.3/

### 2. PTB-XL-Image-17K (Feb 2026)
- **Size**: 17,271 synthetic 12-lead ECG images
- **Features**:
  - Pixel-level segmentation masks
  - Ground truth time-series signals
  - YOLO-format bounding boxes
  - Comprehensive metadata
- **Configurations**: 25/50 mm/s, 5/10 mm/mV, 500 Hz
- **Link**: https://github.com/naqchoalimehdi/PTB-XL-Image-17K

### 3. ECG-Image-Database (PhysioNet 2024)
- **Size**: 35,595 images
- **Content**: Synthetic and real paper ECGs
- **Artifacts**: Wrinkles, stains, perspective shifts, mold
- **Sources**: PTB-XL (977) + Emory Healthcare (1,000)
- **Link**: https://physionet.org/content/ecg-image-database/1.0.0/

### 4. Open-ECG-Digitizer Development Dataset
- **Size**: 37,191 images (public) + 1,596 clinical
- **Features**: Pixel-level annotations for traces, grid, background
- **Link**: https://huggingface.co/datasets/Ahus-AIM/Open-ECG-Digitizer-Development-Dataset

### 5. ECG-Image-Kit
- **Purpose**: Synthetic ECG image generation
- **Features**: Realistic artifacts, customizable parameters
- **Usage**: Data augmentation and training
- **Link**: https://github.com/alphanumericslab/ecg-image-kit

---

## Open Source Tools and Repositories

### 1. ECG-Digitiser (PhysioNet 2024 Winner)
- **Authors**: Krones et al. (SignalSavants)
- **Approach**: Hough Transform + nnU-Net + Vectorization
- **Performance**: SNR 17.02 dB (CV), 12.15 Challenge score
- **Link**: https://github.com/felixkrones/ECG-Digitiser

### 2. Open-ECG-Digitizer
- **Authors**: Stenhede et al. (Ahus-AIM)
- **Approach**: Modular pipeline with U-Net segmentation
- **Performance**: Mean SNR 19.65 dB on scanned papers
- **Features**: Perspective correction, grid detection, lead identification
- **Link**: https://github.com/Ahus-AIM/Open-ECG-Digitizer
- **Paper**: https://doi.org/10.1038/s41746-025-02327-1

### 3. ECGtizer
- **Authors**: Lence et al.
- **Approach**: Automated lead detection + 3 extraction algorithms + DL reconstruction
- **Features**: Signal recovery from degraded papers, completion model
- **Link**: https://github.com/UMMISCO/ecgtizer

### 4. ECG-Image-Kit
- **Authors**: Shivashankara et al. (Emory/Georgia Tech)
- **Purpose**: Synthetic image generation toolbox
- **Features**: Realistic artifacts, grid customization
- **Paper**: https://doi.org/10.1088/1361-6579/ad4954
- **Link**: https://github.com/alphanumericslab/ecg-image-kit

### 5. ECG-Digitization (Ritika Jha)
- **Approach**: Traditional image processing pipeline
- **Features**: Grid detection, signal extraction
- **Link**: https://github.com/ritikajha/ECG-Digitization

### 6. ecg-digitize (Tereshchenko Lab)
- **Approach**: Library and command-line tool
- **Link**: https://github.com/Tereshchenkolab/ecg-digitize

---

## References and Links

### Key Papers

1. **Krones et al. (2024)** - PhysioNet 2024 Winner
   - *Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts*
   - arXiv: https://arxiv.org/abs/2410.14185
   - GitHub: https://github.com/felixkrones/ECG-Digitiser

2. **Stenhede et al. (2025)**
   - *Digitizing Paper ECGs at Scale: An Open-Source Algorithm for Clinical Research*
   - arXiv: https://arxiv.org/abs/2510.19590
   - DOI: https://doi.org/10.1038/s41746-025-02327-1
   - GitHub: https://github.com/Ahus-AIM/Open-ECG-Digitizer

3. **Karbasi et al. (2025)**
   - *Deep Learning-Based Digitization of Overlapping ECG Images with Open-Source Python Code*
   - arXiv: https://arxiv.org/abs/2506.10617
   - Focus: Handling overlapping signals

4. **Lence et al. (2024)**
   - *ECGtizer: a fully automated digitizing and signal recovery pipeline for electrocardiograms*
   - arXiv: https://arxiv.org/abs/2412.12139
   - GitHub: https://github.com/UMMISCO/ecgtizer

5. **Shivashankara et al. (2024)**
   - *ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization*
   - DOI: https://doi.org/10.1088/1361-6579/ad4954
   - GitHub: https://github.com/alphanumericslab/ecg-image-kit

6. **Reyna et al. (2024)**
   - *ECG-Image-Database: A Dataset of ECG Images with Real-World Imaging and Scanning Artifacts*
   - arXiv: https://arxiv.org/abs/2409.16612
   - PhysioNet: https://physionet.org/content/ecg-image-database/1.0.0/

7. **Mehdi (2026)**
   - *PTB-XL-Image-17K: A Large-Scale Synthetic ECG Image Dataset with Comprehensive Ground Truth for Deep Learning-Based Digitization*
   - arXiv: https://arxiv.org/abs/2602.07446
   - GitHub: https://github.com/naqchoalimehdi/PTB-XL-Image-17K

8. **Pazos-Santomé et al. (2024)**
   - *Automated Optical Reading of Scanned ECGs*
   - arXiv: https://arxiv.org/abs/2408.11425

### Challenge and Competition Resources

9. **PhysioNet Challenge 2024**
   - Website: https://moody-challenge.physionet.org/2024/
   - Paper: *Digitization and Classification of ECG Images: The George B. Moody PhysioNet Challenge 2024*
   - DOI: https://doi.org/10.22489/CinC.2024.296

### Related Research Areas

10. **ECG Classification**
    - Deep learning for arrhythmia detection
    - Multi-label classification
    - PhysioNet Challenge 2021 (related)

11. **Signal Processing**
    - R-peak detection
    - Waveform delineation
    - Noise reduction

12. **Medical Image Analysis**
    - Document analysis
    - Grid detection
    - Line extraction

---

## Summary of Key Findings

### State-of-the-Art Performance (as of 2025)

| Method | SNR (dB) | Dataset | Key Innovation |
|--------|----------|---------|----------------|
| SignalSavants (2024) | 17.02 | PTB-XL synthetic | Hough + nnU-Net + Vectorization |
| Open-ECG-Digitizer | 19.65 | Akershus Hospital | Modular pipeline, perspective correction |
| ECGtizer | ~16-17 | JOCOVID + PTB-XL | Signal completion/reconstruction |
| Karbasi et al. | - | Overlapping signals | Overlap handling with U-Net |

### Key Trends

1. **Segmentation-based approaches dominate**: U-Net variants show best performance
2. **Hybrid pipelines**: Combining traditional CV (Hough) with deep learning
3. **Open source**: Most recent work is openly available
4. **Robustness focus**: Handling real-world artifacts (wrinkles, stains, perspective)
5. **Signal completion**: Recovery of partial/degraded signals
6. **Standardization**: Emergence of standard datasets and evaluation metrics

### Open Challenges

1. **Overlapping signals**: Still challenging, especially in dense ECG layouts
2. **Extreme artifacts**: Severely degraded or stained papers
3. **Multi-format support**: Different ECG paper layouts and manufacturers
4. **Real-time processing**: Speed optimization for clinical use
5. **Uncertainty quantification**: Confidence estimates for digitized signals

---

*This summary was compiled on March 14, 2026, based on research papers from 2023-2026, with a focus on approaches published around and after the PhysioNet Challenge 2024.*
