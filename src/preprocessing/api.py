import numpy as np
from typing import Union

from .config import PreprocessingConfig
from .models import PreprocessedECG, Diagnostics
from .io import load_image
from .illumination import apply_illumination_normalization
from .enhance import to_agnostic_grayscale, standardize_resolution, isolate_grid_lines
from .rotation import estimate_rotation_angle, apply_rotation


def preprocess_ecg_image(
    image_source: Union[str, np.ndarray], config: PreprocessingConfig = None
) -> PreprocessedECG:
    """
    Execute the dual-stream standalone ECG preprocessing pipeline.

    This loads an untouched high-res RGB representation for the output,
    forks a scaled, highly filtered 1D structural skeleton specifically
    for calculating robust alignment geometries, and then cleanly applies
    that calculated target translation to the pristine original.
    """
    if config is None:
        config = PreprocessingConfig()

    # ================== STREAM A ==================
    # 1. Standardize the data payload output cache
    original_rgb_image = load_image(image_source)
    original_shape = original_rgb_image.shape

    # ================== STREAM B ==================
    # 2. Extract structural data stream for geometry calculation
    calc_image = to_agnostic_grayscale(
        original_rgb_image, method=config.color_agnostic_method
    )

    # Scale to ensure global threshold algorithms apply proportionately
    calc_image = standardize_resolution(
        calc_image, target_long_edge=config.calculation_scale_resolution
    )
    calculation_shape = calc_image.shape

    # 3. Optional illumination normalization (runs much faster on 1D calc array)
    illumination_applied = False
    illumination_method_used = None
    if config.enable_illumination_normalization:
        calc_image = apply_illumination_normalization(calc_image, config)
        illumination_applied = True
        illumination_method_used = config.illumination_method

    # 4. Optional strictly-orthogonal grid extraction to ignore text/squiggly signal leads
    grid_enhancement_applied = False
    if config.enable_grid_enhancement:
        calc_image = isolate_grid_lines(
            calc_image, morphological_length=config.grid_morphology_length
        )
        grid_enhancement_applied = True

    # 5. Extract domain orientation off our 1D isolated calc plane
    angle, lines_detected, lines_filtered, parallel_lines = estimate_rotation_angle(
        calc_image, config
    )

    # ================== CROSSOVER ==================
    # 6. Apply calculated geometrical correction back to our pristine untouched Stream A
    rotated_image = apply_rotation(original_rgb_image, angle)

    # 7. Formulate the response with debug provenance
    fallback_used = angle == config.default_rotation_angle and parallel_lines == 0
    fallback_reason = (
        "No contiguous parallel lines found in bounds." if fallback_used else None
    )
    diagnostics = Diagnostics(
        original_shape=original_shape,
        calculation_shape=calculation_shape,
        lines_detected=lines_detected,
        lines_filtered=lines_filtered,
        parallel_lines_kept=parallel_lines,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        illumination_applied=illumination_applied,
        illumination_method_used=illumination_method_used,
        grid_enhancement_applied=grid_enhancement_applied,
    )

    return PreprocessedECG(
        image=rotated_image, rotation_angle=angle, diagnostics=diagnostics
    )
