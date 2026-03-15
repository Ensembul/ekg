import cv2
import numpy as np
import math
from typing import Tuple, Optional
from PIL import Image

from .models import PreprocessingConfig
from .errors import RotationEstimationError


def is_within_x_degrees_of_horizontal(theta: float, degree_window: float) -> bool:
    """Check if the line angle is within X degrees of horizontal (90 degrees / pi/2)."""
    theta_degrees = math.degrees(theta)
    deviation_from_horizontal = abs(90.0 - theta_degrees)
    return deviation_from_horizontal < degree_window


def get_lines(image: np.ndarray, config: PreprocessingConfig) -> Optional[np.ndarray]:
    """Detect lines in the image using Canny edge detection and Hough transform."""

    # Convert RGB semantics to Grayscale gracefully if it isn't already 1D.
    if len(image.shape) == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply the Canny edge detector to find edges in the image
    edges = cv2.Canny(
        gray_image,
        int(config.canny_threshold1),
        int(config.canny_threshold2),
        apertureSize=config.canny_aperture_size,
    )

    # Use HoughLines to find lines in the edge-detected image
    lines = cv2.HoughLines(
        edges,
        config.hough_rho_resolution,
        np.pi / 180.0,
        config.hough_threshold,
        None,
        0,
        0,
    )
    return lines


def filter_lines(
    lines: Optional[np.ndarray], config: PreprocessingConfig
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Filter the lines to isolate those indicating the rotation angle.
    Returns:
        Tuple containing:
        - The filtered parallel lines array (or None)
        - Count of lines initially within the horizontal window
        - Count of lines remaining after parallelism check
    """
    if lines is None:
        return None, 0, 0

    parallelism_radian = np.deg2rad(config.parallelism_window)
    filtered_lines = []

    # Filter lines to be within the degree window of horizontal
    for line in lines:
        for rho, theta in line:
            if is_within_x_degrees_of_horizontal(theta, config.degree_window):
                filtered_lines.append((rho, theta))

    lines_filtered_count = len(filtered_lines)

    # Further filter lines based on parallelism
    parallel_lines = []
    if lines_filtered_count > 0:
        # Optimization: Don't compare ALL lines if there are too many. It's O(N^2).
        # We can sort by theta and do a faster check, or just cap the limits.
        # Since we just want to find lines with sufficient parallel supporters:
        max_lines_to_check = 500
        if lines_filtered_count > max_lines_to_check:
            # Randomly sample if we have an absurd amount of lines to prevent N^2 freeze
            import random

            random.seed(42)
            lines_to_check = random.sample(filtered_lines, max_lines_to_check)
        else:
            lines_to_check = filtered_lines

        for rho, theta in lines_to_check:
            count = 0
            for comp_rho, comp_theta in lines_to_check:
                # Check angular distance considering circular wraparound of pi
                if (
                    abs(theta - comp_theta) < parallelism_radian
                    or abs((theta - comp_theta) - math.pi) < parallelism_radian
                ):
                    count += 1

            if count >= config.parallelism_count:
                parallel_lines.append((rho, theta))

    parallel_lines_kept_count = len(parallel_lines)

    if parallel_lines_kept_count == 0:
        return None, lines_filtered_count, 0

    # Return in HoughLines format (N, 1, 2)
    formatted_parallel_lines = np.array(parallel_lines)[:, np.newaxis, :]
    return formatted_parallel_lines, lines_filtered_count, parallel_lines_kept_count


def get_median_degrees(lines: np.ndarray) -> float:
    """Get the median angle of the lines in degrees mapping directly to geometric rotation required."""
    lines_squeezed = lines[:, 0, :]
    # The mathematical transform identical to the reference codebase:
    line_angles = [-(90.0 - math.degrees(theta)) for _, theta in lines_squeezed]
    return round(float(np.median(line_angles)), 4)


def estimate_rotation_angle(
    image: np.ndarray, config: PreprocessingConfig
) -> Tuple[float, int, int, int]:
    """
    Estimate the rotation angle needed to dewarp the ECG grid lines.
    Returns:
        Tuple mapping to (angle, total_lines_detected, filtered_horizontal, parallel_found).
    """
    lines = get_lines(image, config)
    total_detected = len(lines) if lines is not None else 0

    filtered_lines, filtered_horizontal, parallel_found = filter_lines(lines, config)

    if filtered_lines is None:
        if config.fail_on_missing_lines:
            raise RotationEstimationError(
                f"Failed to find rotation angle. Detected {total_detected} lines, "
                f"{filtered_horizontal} horizontal, {parallel_found} parallel."
            )
        return (
            config.default_rotation_angle,
            total_detected,
            filtered_horizontal,
            parallel_found,
        )

    rot_angle = get_median_degrees(filtered_lines)

    if math.isnan(rot_angle):
        if config.fail_on_missing_lines:
            raise RotationEstimationError("Median rotation angle is NaN.")
        rot_angle = config.default_rotation_angle

    return rot_angle, total_detected, filtered_horizontal, parallel_found


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate the image by the given angle.
    Parity note: Uses PIL's rotation (mirrors torchvision's underlying functional implementation).
    """
    if angle == 0.0:
        return image.copy()

    # torchvision.transforms.functional.rotate uses PIL Image.rotate internally.
    pil_image = Image.fromarray(image)
    rotated_pil = pil_image.rotate(
        angle, resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0)
    )
    return np.array(rotated_pil)
