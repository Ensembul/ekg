#!/usr/bin/env python3
"""
ECG Digitization Pipeline — CLI Entry Point

Usage:
    # Validate on training set (first 100 images)
    python main.py --mode train --input_dir ../data/train/ --n_samples 100

    # Generate test submission
    python main.py --mode test --input_dir ../data/test/ --output submission.npz
"""

import argparse
import sys
import os

# Ensure the_last_hope directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import ECGDigitizer
from validate import run_validation


def main():
    parser = argparse.ArgumentParser(description="ECG Digitization Pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        required=True,
        help="'train' to validate against ground truth, 'test' to generate submission",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing ECG images",
    )
    parser.add_argument(
        "--output",
        default="submission.npz",
        help="Output path for submission file (test mode only)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of training images to validate (train mode only)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        print(f"Validating on {args.n_samples} training images...")
        scores = run_validation(args.input_dir, n_samples=args.n_samples)
        if scores:
            import numpy as np

            all_totals = [
                s["total_score"]
                for record_scores in scores
                for s in record_scores.values()
            ]
            avg = np.mean(all_totals)
            if avg >= 85:
                print(f"\nTarget achieved ({avg:.1f} >= 85)! Ready for submission.")
            else:
                print(f"\nBelow target ({avg:.1f} < 85). Tune parameters.")
    else:
        print(f"Processing test images from {args.input_dir}...")
        digitizer = ECGDigitizer()
        digitizer.process_dataset(args.input_dir, args.output)


if __name__ == "__main__":
    main()
