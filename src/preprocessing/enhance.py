import cv2
import numpy as np


def to_agnostic_grayscale(image: np.ndarray, method: str = "lab") -> np.ndarray:
    """
    Converts an RGB image to a structural intensity channel.
    This process optimally isolates intensity contrasts (like dark grids on white paper)
    and ignores hue tints (e.g., pink/red paper backgrounds masking red/black ink).

    Args:
        image: Original RGB numpy array of shape (H, W, 3).
        method: Color space logic to use. "lab" for L-channel or "hsv" for V-channel.

    Returns:
        np.ndarray: A 1D array of shape (H, W) representing structurally agnostic lightness/value.
    """
    if len(image.shape) == 2:
        return image  # Already grayscale

    if method.lower() == "hsv":
        # Convert to HSV space: Hue, Saturation, Value (Brightness)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        _, _, v_channel = cv2.split(hsv)
        return v_channel
    else:
        # Default to LAB space: L (Lightness), A (Green-Red), B (Blue-Yellow)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, _, _ = cv2.split(lab)
        return l_channel


def standardize_resolution(
    image: np.ndarray, target_long_edge: int = 2000
) -> np.ndarray:
    """
    Scales the image so its longest edge explicitly hits the target parameter,
    preserving aspect ratio. Ensures pixel-based thresholds behave consistently.

    Args:
        image: Numpy array of any dimensionality.
        target_long_edge: Integer target length for the longest edge.

    Returns:
        np.ndarray: Scaled array.
    """
    h, w = image.shape[:2]
    max_edge = max(h, w)

    if max_edge == target_long_edge:
        return image

    scale_factor = target_long_edge / max_edge
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def isolate_grid_lines(
    gray_image: np.ndarray, morphological_length: int = 50
) -> np.ndarray:
    """
    Applies directional morphological operations to extract only long, orthogonally straight lines
    from the array, effectively eliminating sparse text, noise, and squiggly signal traces.

    Args:
        gray_image: 1D grayscale numpy array (H, W).
        morphological_length: Length of grid kernels in pixels. Lower catches tighter grids,
                              higher strictly catches long, unbroken lines.

    Returns:
        np.ndarray: A strictly high-contrast array with isolated grid structures.
    """
    # Morphological line detection operates on white features with black backgrounds.
    # Standard document images are black ink on white background, so we invert.
    inverted_image = cv2.bitwise_not(gray_image)

    # Extract continuous horizontal structures
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (morphological_length, 1)
    )
    horizontal_lines = cv2.morphologyEx(
        inverted_image, cv2.MORPH_OPEN, horizontal_kernel
    )

    # Extract continuous vertical structures
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, morphological_length)
    )
    vertical_lines = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, vertical_kernel)

    # Combine orthogonal frames back together
    isolated_grids = cv2.add(horizontal_lines, vertical_lines)

    # Invert back to normal document layout (dark traces on light backgrounds)
    return cv2.bitwise_not(isolated_grids)
