from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class PreprocessingConfig:
    """Configuration for ECG image preprocessing algorithms."""

    # Hough transform parameters
    hough_threshold: int = 1200  # 1200
    hough_rho_resolution: float = 1.0  # pixels

    # Canny edge detection parameters
    canny_threshold1: float = 50  # 50
    canny_threshold2: float = 100  # 150
    canny_aperture_size: int = 3

    # Line filtering parameters
    degree_window: float = 30.0  # degrees from horizontal
    parallelism_count: int = 3
    parallelism_window: float = 2.0  # degrees

    # Behavior parameters
    fail_on_missing_lines: bool = False
    default_rotation_angle: float = 0.0

    # Illumination normalization parameters
    enable_illumination_normalization: bool = False
    illumination_method: str = "morphology"  # Options: "morphology", "clahe"
    illumination_morph_kernel_size: int = 25
    illumination_clahe_clip_limit: float = 2.5
    illumination_clahe_tile_size: int = 8

    # Enhance and scale calculation parameters
    enable_grid_enhancement: bool = False
    color_agnostic_method: str = "lab"  # Options: "lab" (Lightness) or "hsv" (Value)
    calculation_scale_resolution: int = (
        2000  # Scale longest edge to this for consistent Hough thresholds
    )
    grid_morphology_length: int = (
        50  # Size of kernel to isolate long straight grid lines
    )


@dataclass
class Diagnostics:
    """Metadata and diagnostic information generated during preprocessing."""

    original_shape: Tuple[int, int, int]
    calculation_shape: Tuple[int, int, int]
    lines_detected: int
    lines_filtered: int
    parallel_lines_kept: int
    fallback_used: bool
    fallback_reason: Optional[str] = None
    illumination_applied: bool = False
    illumination_method_used: Optional[str] = None
    grid_enhancement_applied: bool = False


@dataclass
class PreprocessedECG:
    """The result of the ECG preprocessing pipeline."""

    image: np.ndarray  # The (H, W, 3) rotated image array
    rotation_angle: float  # Angle in degrees
    diagnostics: Diagnostics
