# Agnostic ECG Preprocessing Implementation Plan

## Objective
Extract and export the pre-segmentation processing logic covering image loading, rotation-angle estimation, and dewarping into a standalone, framework-agnostic package at `./preprocessing`.

## Constraints & Rules
- **No Torch Dependency**: The package must not import `torch` or `torchvision`. Operations should execute exclusively via NumPy, OpenCV (cv2), and Python standard library primitives.
- **Isolate from Legacy Code**: Do NOT import any configurations, utilities, or helpers from the `ECG-digitiser` package.
- **Agnostic Boundary**: Stop execution *immediately* prior to the segmentation inference. Generating sub-process shell commands or writing formatted tensors destined explicitly for a targeted framework like nnUNet belongs in an external adapter, not the core preprocessing library.

## Step-by-Step Implementation

### Phase 1: Package Scaffolding & API Contracts
1. Create the package directory `./preprocessing`.
2. Define the exact input/output boundaries using Python `dataclasses`.
3. Create `config.py` holding algorithm thresholds (e.g., Canny edges, Hough lines windows).
4. Create lightweight custom exceptions in `errors.py`.

### Phase 2: Building Core Primitives
1. Create `io.py`: Implement robust image loading converting 1-channel grayscale or 4-channel RGBA into standard 3-channel BGR/RGB matrices via OpenCV. Do not use PyTorch/Torchvision.
2. Create `rotation.py`:
   - Implement `get_lines` via `cv2.Canny` and `cv2.HoughLines`.
   - Implement `filter_lines` checking horizontality window (±30°) and parallelism count.
   - Implement `get_median_degrees` ensuring fallbacks for missing lines.
   - Implement `apply_rotation` mapping angles through an affine transform (`cv2.warpAffine`) mirroring the legacy behavior, avoiding PIL interpolations if possible.

### Phase 3: Public API Orchestrator (Facade)
1. Create `api.py` serving as the main entry point (e.g., `preprocess_ecg_image`).
2. Wire I/O processing pipeline → run Angle Estimation → run Dewarping rotation → return dataclass output encompassing the clean matrix and rotation metadata.

### Phase 4: Downstream Adaptor Hooks
1. Set up an `adapters/` subdirectory housing a generic Segmenter protocol.
2. Ensure the main `preprocess_ecg_image` doesn't enforce serialization conventions that dictate how down-streamers receive the image.

### Phase 5: Verification & Tests
1. Add golden tests verifying numeric parity on angle calculations against the legacy implementation.
2. Assert 0-dependency guard rails making sure `import torch` does not inadvertently slip in.
