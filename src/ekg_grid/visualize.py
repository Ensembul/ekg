import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


def visualize_intersections(
    image: np.ndarray,
    intersections: List[Tuple[int, int]],
    point_color: Tuple[int, int, int] = (0, 255, 0),
    point_radius: int = 3,
    alpha: float = 0.8,
) -> np.ndarray:
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        vis_image = image.copy()

    overlay = vis_image.copy()
    for x, y in intersections:
        cv2.circle(overlay, (x, y), point_radius, point_color, -1)

    result = cv2.addWeighted(vis_image, alpha, overlay, 1 - alpha, 0)
    return result


def plot_intersections_matplotlib(
    image: np.ndarray,
    intersections: List[Tuple[int, int]],
    title: str = "Detected Grid Intersections",
    point_size: int = 10,
):
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = image.squeeze()

    plt.figure(figsize=(12, 10))
    plt.imshow(image)

    if intersections:
        xs, ys = zip(*intersections)
        plt.scatter(xs, ys, c="red", s=point_size, marker=".", alpha=0.7)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def save_visualization(
    image: np.ndarray,
    intersections: List[Tuple[int, int]],
    output_path: str,
    point_color: Tuple[int, int, int] = (0, 255, 0),
    point_radius: int = 3,
):
    if len(image.shape) == 3 and image.shape[2] == 4:
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        vis_image = image.copy()

    for x, y in intersections:
        cv2.circle(vis_image, (x, y), point_radius, point_color, -1)

    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
