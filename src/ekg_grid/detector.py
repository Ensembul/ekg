import cv2
import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import maximum_filter


class GridDetector:
    def __init__(
        self,
    ):
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        # Otsu's threshold - invert so grid lines are bright
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary

    def find_grid_lines(self, binary: np.ndarray) -> Tuple[List[int], List[int]]:
        # Vertical projection - sum of bright pixels per column
        proj_v = np.sum(binary > 0, axis=0).astype(float)

        if len(proj_v) == 0:
            return [], []

        # Use scipy for efficiency
        peaks_v = maximum_filter(proj_v, size=20)
        # Use 70th percentile - works reasonably across different images
        threshold_v = np.percentile(proj_v, 70)
        local_max_v = (proj_v == peaks_v) & (proj_v > threshold_v)
        vertical_lines = np.where(local_max_v)[0].tolist()

        # Horizontal projection
        proj_h = np.sum(binary > 0, axis=1).astype(float)

        if len(proj_h) == 0:
            return [], []

        peaks_h = maximum_filter(proj_h, size=20)
        threshold_h = np.percentile(proj_h, 70)
        local_max_h = (proj_h == peaks_h) & (proj_h > threshold_h)
        horizontal_lines = np.where(local_max_h)[0].tolist()

        return vertical_lines, horizontal_lines

    def detect(self, image: np.ndarray) -> List[Tuple[int, int]]:
        binary = self.preprocess(image)
        vertical_lines, horizontal_lines = self.find_grid_lines(binary)

        # Compute all pairwise intersections
        intersections = []
        for y in horizontal_lines:
            for x in vertical_lines:
                intersections.append((x, y))

        return intersections


def process_image(
    image_path: str, output_path: Optional[str] = None
) -> List[Tuple[int, int]]:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    detector = GridDetector()
    intersections = detector.detect(image)

    if output_path:
        vis_image = image.copy()
        for x, y in intersections:
            cv2.circle(vis_image, (x, y), 3, (0, 255, 0), -1)
        cv2.imwrite(output_path, vis_image)

    return intersections
