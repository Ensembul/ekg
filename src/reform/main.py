"""
Batch ECG digitization → submission.npz

Usage
-----
  python main.py ../data/test/ -o submission.npz
  python main.py ../data/test/ -o submission.npz --binarize color --method full
"""

import argparse
from pathlib import Path

import numpy as np
from deskew import determine_skew
from PIL import Image
from tqdm import tqdm

from adapter import ALL_LEAD_NAMES, adapt
from extract import OUTPUT_SAMPLES, digitize


def deskew(pil_image: Image.Image) -> Image.Image:
    """Straighten a scanned/photographed ECG image."""
    img_np = np.array(pil_image)
    if img_np.ndim == 3 and img_np.shape[2] >= 3:
        gray = np.mean(img_np[:, :, :3], axis=2).astype(np.uint8)
    else:
        gray = img_np

    angle = determine_skew(gray, max_angle=45.0, num_peaks=20)
    if angle is None or abs(angle) < 0.1:
        return pil_image

    return pil_image.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=False,
        fillcolor=(255, 255, 255),
    )


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    image_files = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    submission: dict[str, np.ndarray] = {}
    failed = 0

    for image_path in tqdm(image_files, desc="Digitizing"):
        record_name = image_path.stem
        try:
            pil_image = Image.open(image_path).convert("RGB")
            if args.deskew:
                pil_image = deskew(pil_image)
            leads = digitize(
                pil_image,
                binarize=args.binarize,
                extract_method=args.method,
                hp_cutoff=args.hp if args.hp else None,
                lp_cutoff=None if args.no_lp else 40.0,
                mask_text_first=args.mask_text,
            )
            submission.update(adapt(record_name, leads))
        except Exception as e:
            failed += 1
            print(f"FAILED {record_name}: {e}")
            for lead_name in ALL_LEAD_NAMES:
                submission[f"{record_name}_{lead_name}"] = np.zeros(
                    OUTPUT_SAMPLES, dtype=np.float16
                )

    print(f"Submission keys ({len(submission)}): {sorted(submission.keys())[:15]} ...")
    np.savez_compressed(args.output, **submission)

    n_records = len(image_files)
    n_signals = len(submission)
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(
        f"Done: {n_records} records, {n_signals} signals, "
        f"{failed} failures → {args.output} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG digitization → submission.npz")
    parser.add_argument("input_dir", help="Folder containing ECG images")
    parser.add_argument("-o", "--output", default="submission.npz")
    parser.add_argument(
        "--binarize",
        default="auto",
        choices=["auto", "color", "canny", "otsu", "adaptive"],
    )
    parser.add_argument(
        "--method",
        default="twopass",
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
        help="Whole-image text masking (default: none; per-block masking is always on for auto)",
    )
    parser.add_argument(
        "--deskew", action="store_true",
        help="Apply deskew rotation correction",
    )
    run(parser.parse_args())
