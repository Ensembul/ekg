"""
Adapter: digitize() output → submission.npz

Converts the dict[lead_name → 5000-sample float64 array] from digitize()
into the competition submission format:
  - Keys: {record_name}_{lead_name}  (e.g. "ecg_test_0001_I")
  - Values: np.float16, shape (5000,)
  - NaN/Inf replaced with 0.0
  - Saved via np.savez_compressed
"""

from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from extract import LEAD_GRID, OUTPUT_SAMPLES, digitize

ALL_LEAD_NAMES = [name for row in LEAD_GRID for name in row]


def to_submission_dict(
    record_name: str,
    leads: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Convert a single record's digitize() output to submission key/value pairs.

    Missing leads are zero-filled. NaN/Inf values are replaced with 0.
    """
    result = {}
    for lead_name in ALL_LEAD_NAMES:
        key = f"{record_name}_{lead_name}"
        if lead_name in leads:
            signal = np.nan_to_num(leads[lead_name], nan=0.0, posinf=0.0, neginf=0.0)
        else:
            signal = np.zeros(OUTPUT_SAMPLES)
        result[key] = signal.astype(np.float16)
    return result


def generate_submission(
    input_dir: str,
    output_path: str = "submission.npz",
    binarize: str = "auto",
    extract_method: str = "twopass",
    hp_cutoff: float | None = None,
    lp_cutoff: float | None = 40.0,
    mask_text_first: bool | str = False,
) -> None:
    """
    Process all images in input_dir and produce submission.npz.
    """
    image_files = sorted(
        p
        for p in Path(input_dir).iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    submission: dict[str, np.ndarray] = {}
    failed = 0

    for image_path in tqdm(image_files, desc="Digitizing ECGs"):
        record_name = image_path.stem
        try:
            pil_image = Image.open(image_path).convert("RGB")
            leads = digitize(
                pil_image,
                binarize=binarize,
                extract_method=extract_method,
                hp_cutoff=hp_cutoff,
                lp_cutoff=lp_cutoff,
                mask_text_first=mask_text_first,
            )
            submission.update(to_submission_dict(record_name, leads))
        except Exception as e:
            print(f"FAILED {record_name}: {e}")
            failed += 1
            for lead_name in ALL_LEAD_NAMES:
                submission[f"{record_name}_{lead_name}"] = np.zeros(
                    OUTPUT_SAMPLES, dtype=np.float16
                )

    np.savez_compressed(output_path, **submission)
    print(f"Saved {len(submission)} signals to {output_path} ({failed} failures)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate submission.npz from ECG images"
    )
    parser.add_argument("input_dir", help="Directory containing ECG images")
    parser.add_argument("-o", "--output", default="submission.npz")
    parser.add_argument(
        "--binarize", default="auto",
        choices=["auto", "color", "canny", "otsu", "adaptive"],
    )
    parser.add_argument(
        "--method", default="twopass",
        choices=["twopass", "lazy", "full", "fragmented", "viterbi"],
    )
    parser.add_argument(
        "--hp", type=float, default=None,
        help="Enable high-pass filter at given frequency (Hz)",
    )
    parser.add_argument("--no-lp", action="store_true", help="Disable low-pass filter")
    parser.add_argument(
        "--mask-text",
        default="none",
        choices=["morphological", "tesseract", "none"],
        help="Whole-image text masking (default: none; per-block masking always on for auto)",
    )
    args = parser.parse_args()

    generate_submission(
        args.input_dir,
        args.output,
        binarize=args.binarize,
        extract_method=args.method,
        hp_cutoff=args.hp if args.hp else None,
        lp_cutoff=None if args.no_lp else 40.0,
        mask_text_first=args.mask_text,
    )
