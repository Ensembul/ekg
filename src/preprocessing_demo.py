import os
import argparse
from PIL import Image

from preprocessing import preprocess_ecg_image


def main():
    parser = argparse.ArgumentParser(
        description="Demo script for standalone ECG image preprocessing."
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to the input ECG image."
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        required=True,
        help="Directory to save the preprocessed image.",
    )
    # Illumination Normalization Arguments
    parser.add_argument(
        "--illumination", action="store_true", help="Enable illumination normalization."
    )
    parser.add_argument(
        "--illumination-method",
        type=str,
        choices=["morphology", "clahe"],
        default="morphology",
        help="Method for illumination normalization.",
    )

    parser.add_argument(
        "--grid-enhancement",
        action="store_true",
        help="Enable morphological grid extraction prior to Canny detection.",
    )
    parser.add_argument(
        "--color-method",
        type=str,
        choices=["lab", "hsv"],
        default="lab",
        help="Color space extraction method for structural agnosticism. Default 'lab' uses Lightness. 'hsv' uses Value.",
    )

    args = parser.parse_args()

    input_path = args.input
    save_dir = args.save_dir

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.")
        return

    print(f"Processing image: {input_path}")

    from preprocessing.models import PreprocessingConfig

    # Setup optional config overrides based on argparse
    config_overrides = {"color_agnostic_method": args.color_method}
    if args.color_method != "lab":
        print(f"-> Using structural color isolation: {args.color_method.upper()}")

    if args.illumination:
        config_overrides["enable_illumination_normalization"] = True
        config_overrides["illumination_method"] = args.illumination_method
        print(f"-> Using illumination normalization: {args.illumination_method}")

    if args.grid_enhancement:
        config_overrides["enable_grid_enhancement"] = True
        print("-> Using morphological grid enhancement")

    config = PreprocessingConfig(**config_overrides)

    try:
        # Perform preprocessing via the extracted, torch-free API
        result = preprocess_ecg_image(input_path, config=config)

        # Display the derived angle and diagnostics
        print(f"Rotation angle applied: {result.rotation_angle:.2f} degrees")
        print(f"Diagnostics: {result.diagnostics}")

        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Construct output filename and path
        base_name = os.path.basename(input_path)
        out_name = f"preprocessed_{base_name}"
        out_path = os.path.join(save_dir, out_name)

        # The result.image is an RGB uint8 numpy array (H, W, 3) provided by our IO implementation
        # Convert back to PIL Image and save
        out_image = Image.fromarray(result.image)
        out_image.save(out_path)

        print(f"Successfully saved preprocessed image to: {out_path}")

    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")


if __name__ == "__main__":
    main()
