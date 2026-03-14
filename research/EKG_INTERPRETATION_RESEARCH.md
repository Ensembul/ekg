# EKG/ECG Image Interpretation - Comprehensive Research Document

## Overview

This document provides a comprehensive overview of current approaches, tools, and resources for interpreting EKG readings from physical images (paper ECGs, photos, scans). The field involves two main tasks:

1. **Digitization**: Converting ECG images into digital time-series signals
2. **Classification/Interpretation**: Analyzing the digitized signals (or directly classifying the images) for cardiac conditions

---

## Part 1: ECG Image Digitization (Image → Signal)

### What is ECG Digitization?

Converting paper/photo ECGs into digital time-series data that can be analyzed by ML models or cardiologists.

### Top Open-Source Solutions

#### 1. ECG-Digitiser (PhysioNet Challenge 2024 Winner) ⭐ RECOMMENDED

**Repository**: https://github.com/felixkrones/ECG-Digitiser

- **Winner** of the George B. Moody PhysioNet Challenge 2024
- Combines **Hough Transform** with **deep learning (nnU-Net)**
- Provides pretrained segmentation models
- SNR (Signal-to-Noise Ratio): ~19.65 dB on test data

**Quick Start**:
```bash
git clone https://github.com/felix-krones/ecg-digitiser.git
cd ecg-digitiser
git lfs install
git lfs pull
conda create -n ecgdig python=3.11
conda activate ecgdig
pip install -r requirements.txt

# Run digitization
python -m src.run.digitize -d data_folder -o output_folder
```

**Paper**: https://arxiv.org/abs/2410.14185

---

#### 2. ECG-Image-Kit

**Repository**: https://github.com/alphanumericslab/ecg-image-kit

- Comprehensive toolkit for **synthesis, analysis, and digitization** of ECG images
- BSD 3-Clause License
- Generates synthetic ECG images from PTB-XL dataset
- Includes digitization pipelines with U-Net based approaches

**Features**:
- Synthetic ECG image generation
- Image augmentation (noise, wrinkles, stains)
- Grid detection and signal extraction

**Paper**: https://arxiv.org/abs/2307.01946

---

#### 3. ecgtizer

**Repository**: https://github.com/UMMISCO/ecgtizer

- Fully automated digitization of paper/PDF ECGs
- Handles various ECG formats and artifacts

---

#### 4. Tereshchenkolab/ecg-digitize

**Repository**: https://github.com/Tereshchenkolab/ecg-digitize

- Library and command-line tool for digitizing ECG images
- Simple to use for basic digitization tasks

---

#### 5. paper-ecg (OSU Capstone)

**Repository**: https://github.com/Tereshchenkolab/paper-ecg

- 57 stars, well-maintained
- Classic approach with deep learning enhancements
- Good for learning purposes

---

#### 6. Deep Learning-Based Digitization (U-Net)

**Repository**: https://github.com/masoudrahimi39/ECG-code

- Two-stage pipeline: U-Net segmentation + signal extraction
- Handles **overlapping ECG signals** (common challenge)
- IoU: 0.87 for segmentation
- MSE: 0.0010, Pearson: 0.9644 on non-overlapping signals

**Paper**: https://arxiv.org/html/2506.10617v1

---

#### 7. SCAI-Lab/ecg_digitization

**Repository**: https://github.com/SCAI-Lab/ecg_digitization

- Complete pipeline with 4 stages:
  1. Synthetic image generation
  2. Dataset augmentations
  3. Model training
  4. Digitization pipeline

---

### Key Datasets for ECG Digitization

| Dataset | Description | Link |
|---------|-------------|------|
| **ECG-Image-Database** | 35,595 ECG images with artifacts (PTB-XL + Emory) | https://arxiv.org/abs/2409.16612 |
| **PTB-XL** | 21,797 12-lead ECGs (source for synthetic images) | https://physionet.org/content/ptb-xl/1.0.3/ |
| **PhysioNet Challenge 2024** | Training data for ECG digitization | https://physionet.org/content/ecg-image-challenge/1.0.0/ |

---

### Classical Approaches (Non-DL)

- **Hough Transform**: Detects grid lines and orientation
- **Otsu's thresholding**: Binary segmentation
- **Morphological operations**: Signal extraction
- **Template matching**: Lead detection

Reference: https://www.nature.com/articles/s41598-022-25284-1

---

## Part 2: ECG Image Classification (Image → Diagnosis)

### Direct Image Classification Approaches

These methods classify ECG images directly without explicit digitization.

#### 1. Multimodal LLM for ECG Images (2025)

**Paper**: https://ai.jmir.org/2025/1/e75910/

- Uses GPT-4V and other multimodal models
- Compares with dedicated ECG AI systems
- Detects Myocardial Infarction from ECG images

---

#### 2. ECG-XPLAIM

**Paper**: https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2025.1659971/full

- Explainable locally-adaptive AI model
- Arrhythmia detection from large-scale ECG data

---

#### 3. SimCardioNet (2026)

**Paper**: https://www.nature.com/articles/s41598-026-36932-1

- Hybrid self-supervised + supervised framework
- Multi-scale CNN with residual connections
- Multi-head self-attention
- Pretrained via SimCLR contrastive learning

---

#### 4. DeepECG-Net

**Paper**: https://www.nature.com/articles/s41598-025-07781-1

- Hybrid transformer + CNN for real-time anomaly detection
- Optimized for long-range dependencies

---

### Signal-Based Classification (After Digitization)

Once digitized, use these approaches:

#### 1. ResNet-based Classification

**Repositories**:
- https://github.com/jonchuaenzhe/resnet-ecg-classifier (12-lead, TensorFlow)
- https://github.com/darshjadhav/ECG_ResNet (PyTorch)
- https://github.com/yshanyes/Pytorch-ECG-Classifier-Cinc2020-Official

**Performance** (on 12-lead):
- AFib: Sn 95.0%, Sp 84.7%
- AFl: Sn 90.8%, Sp 98.3%
- SVT: Sn 88.8%, Sp 99.3%
- MI: Sn 94.2%, Sp 91.0%

---

#### 2. PhysioNet/CinC Challenges

- **2020**: Multi-label ECG classification
- **2024**: Image digitization + classification

**Code**: https://github.com/wenh06/cinc2024

---

#### 3. Other Architectures

- **1D CNN**: Basic convolutional networks
- **LSTM/GRU**: Sequential patterns
- **Transformer**: Attention-based models
- **Ensemble**: Combining multiple models

---

## Part 3: Recommended Implementation Pipeline

### Option A: End-to-End (Direct Classification)

```
Image → CNN/ViT → Diagnosis
```

**Pros**: Simpler, no intermediate signal
**Cons**: Less interpretable, needs more data

**Tools**:
- Transfer learning with ImageNet pretrained models (ResNet, EfficientNet)
- Fine-tune on ECG image datasets

---

### Option B: Two-Stage (Digitize → Classify) ⭐ RECOMMENDED

```
Image → Digitization → Signal → 1D CNN/ResNet → Diagnosis
```

**Pros**: 
- State-of-the-art accuracy
- Reusable components
- Interpretable intermediate results
- Can use both signal and image models

**Implementation**:

1. **Digitize** with ECG-Digitiser or ECG-Image-Kit
2. **Classify** with ResNet/ECGNet on the signal

---

## Part 4: Code Examples

### Example 1: Using ECG-Digitiser

```python
# From ECG-Digitiser repository
from pathlib import Path
from src.run.digitize import digitize_ecg

# Single image digitization
result = digitize_ecg(
    image_path="path/to/ecg.jpg",
    model_path="models/nnUNet_results",
    output_dir="output/"
)

# Access digitized signal
signal = result['signal']  # numpy array
leads = result['leads']    # dict of lead signals
```

---

### Example 2: ECG-Image-Kit Digitization

```python
# Basic digitization pipeline
from ecg_image_kit import ECGImageDigitizer

digitizer = ECGImageDigitizer(
    grid_detection=True,
    signal_extraction='adaptive'
)

result = digitizer.digitize('ecg_image.png')
signals = result.signals  # numpy array (num_leads, samples)
```

---

### Example 3: Signal Classification with ResNet (PyTorch)

```python
import torch
import torch.nn as nn

class ResNet1D(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 256, 3)
        self.layer2 = self._make_layer(256, 512, 4, stride=2)
        self.layer3 = self._make_layer(512, 1024, 6, stride=2)
        self.layer4 = self._make_layer(1024, 2048, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(2048, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# Usage
model = ResNet1D(num_classes=5)
# Input: (batch, 12, 1000) for 12-lead ECG with 1000 samples
output = model(input_signal)
```

---

### Example 4: Simple CNN for ECG Classification (Keras)

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_ecg_cnn(input_shape=(1000, 1), num_classes=5):
    model = models.Sequential([
        layers.Conv1D(32, 5, activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(64, 5, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
```

---

## Part 5: Key Papers & References

### Digitization

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Combining Hough Transform and Deep Learning (ECG-Digitiser) | 2024 | PhysioNet Challenge Winner |
| ECG-Image-Kit: Synthetic Generation | 2024 | Toolkit + Dataset |
| Fully-automated Paper ECG Digitisation | 2022 | Deep learning approach |
| Deep Learning-Based Digitization of Overlapping ECG | 2025 | U-Net + overlap handling |

### Classification

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Multi-label Classification with Modified ResNet | 2020 | PhysioNet/CinC 2020 |
| DeepECG-Net: Hybrid Transformer | 2025 | Real-time detection |
| ECG-XPLAIM | 2025 | Explainable AI |
| SimCardioNet | 2026 | Self-supervised + supervised |

---

## Part 6: Practical Recommendations

### For Your EKG Image (ekg.jpg)

Given your image shows a 12-lead ECG printout:

1. **Immediate Classification**: Use direct image classification with pretrained models
2. **Full Pipeline**: 
   - Use ECG-Digitiser to extract signals
   - Apply ResNet/ECGNet for diagnosis

### Key Considerations

- **Image Quality**: Your image is relatively clear but may need preprocessing
- **Grid Removal**: ECG paper grid lines need to be handled
- **Lead Separation**: 12-lead ECGs need individual lead extraction
- **Artifacts**: Photos may have perspective distortion, shadows

### Next Steps

1. Start with ECG-Digitiser for digitization
2. Validate output signals manually
3. Apply classification model to digitized signals
4. Consider ensemble approaches for robustness

---

## Additional Resources

- **PhysioNet**: https://physionet.org/
- **PTB-XL Dataset**: https://physionet.org/content/ptb-xl/1.0.3/
- **ECG Image Database**: https://arxiv.org/abs/2409.16612
- **George B. Moody PhysioNet Challenge 2024**: https://moody-challenge.physionet.org/2024/

---

*Document generated: March 2026*
