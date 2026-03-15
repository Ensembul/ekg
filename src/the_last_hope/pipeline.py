"""
Main pipeline orchestrator.

Wires together every stage: load -> deskew -> binarize -> segment ->
extract -> assemble -> output.
"""

import os
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from calibration import compute_calibration, ALL_LEAD_NAMES, SAMPLES_FULL
from deskew_ecg import deskew_image
from binarize import binarize
from segment import segment_leads
from assemble import assemble_record


class ECGDigitizer:
    """Process ECG images into digitized signals."""

    def process_image(self, image_path: str) -> dict[str, np.ndarray]:
        """
        Process a single ECG image through the full pipeline.

        Returns dict mapping lead name -> 5000-sample float64 array (mV).
        """
        img = Image.open(image_path)

        # Compute calibration from dimensions
        cal = compute_calibration(img.width, img.height)

        # Rotation correction
        img = deskew_image(img)

        # Convert to numpy
        img_array = np.array(img)

        # Binarize (auto-detect colored vs BW)
        binary = binarize(img_array, cal)

        # Segment into lead blocks
        blocks = segment_leads(binary, cal)

        # Assemble all leads
        leads = assemble_record(blocks, cal)

        return leads

    def process_dataset(
        self,
        image_dir: str,
        output_path: str = "submission.npz",
    ) -> None:
        """
        Process an entire directory of ECG images and save submission.npz.
        """
        patterns = ["*.png", "*.jpg", "*.jpeg"]
        image_files = []
        for pat in patterns:
            image_files.extend(glob(os.path.join(image_dir, pat)))
        image_files.sort()

        if not image_files:
            raise FileNotFoundError(f"No images found in {image_dir}")

        submission: dict[str, np.ndarray] = {}
        failed = 0

        for image_path in tqdm(image_files, desc="Digitizing ECGs"):
            record_name = Path(image_path).stem

            try:
                leads = self.process_image(image_path)
                for lead_name, signal in leads.items():
                    key = f"{record_name}_{lead_name}"
                    # Clean signal: replace NaN/Inf with zero
                    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
                    submission[key] = signal.astype(np.float16)
            except Exception as e:
                failed += 1
                print(f"  FAILED {record_name}: {e}")
                for lead_name in ALL_LEAD_NAMES:
                    key = f"{record_name}_{lead_name}"
                    submission[key] = np.zeros(SAMPLES_FULL, dtype=np.float16)

        np.savez_compressed(output_path, **submission)
        total = len(image_files)
        print(f"\nDone: {total - failed}/{total} succeeded, {failed} failed")
        print(f"Submission saved: {output_path} ({len(submission)} entries)")
