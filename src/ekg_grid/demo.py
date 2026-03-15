import os
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ekg_grid.detector import GridDetector
from ekg_grid.visualize import save_visualization


def create_detector_for_image_size(height: int, width: int) -> GridDetector:
    return GridDetector()


def demo():
    train_dir = Path("data/train")
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_files = [
        "ecg_train_0001.png",
        "ecg_train_0002.png",
        "ecg_train_0003.png",
    ]

    for filename in sample_files:
        image_path = train_dir / filename
        print(f"Processing {filename}...")

        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"  Could not load {filename}")
            continue

        detector = create_detector_for_image_size(image.shape[0], image.shape[1])

        # Debug info
        binary = detector.preprocess(image)
        v, h = detector.find_grid_lines(binary)
        print(f"  V lines: {len(v)}, H lines: {len(h)}")

        intersections = detector.detect(image)
        print(f"  Detected {len(intersections)} intersections")

        output_path = output_dir / f"{Path(filename).stem}_detected.png"
        save_visualization(image, intersections, str(output_path))
        print(f"  Saved to {output_path}")

    print("\nNote: This baseline uses projection-based detection.")
    print("For better results, consider the DINOv2 approach from the paper.")


if __name__ == "__main__":
    demo()
