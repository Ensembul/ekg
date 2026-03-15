import unittest
import numpy as np
from preprocessing.api import preprocess_ecg_image
from preprocessing.models import PreprocessingConfig
from preprocessing.errors import ImageLoadError


class TestPreprocessingAPI(unittest.TestCase):
    def setUp(self):
        # Create a dummy image
        self.mock_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Draw some horizontal-ish lines to simulate grid so it passes Hough cleanly
        for y in range(10, 90, 10):
            # Rotated slightly
            for x in range(100):
                self.mock_image[min(y + int(x * 0.05), 99), x] = [255, 255, 255]

    def test_preprocess_from_array(self):
        config = PreprocessingConfig(hough_threshold=10)  # Lowered for tiny test image
        result = preprocess_ecg_image(self.mock_image, config)

        self.assertEqual(result.image.shape, (100, 100, 3))
        self.assertIsInstance(result.rotation_angle, float)
        self.assertIsNotNone(result.diagnostics)
        self.assertFalse(result.diagnostics.illumination_applied)
        self.assertIsNone(result.diagnostics.illumination_method_used)

    def test_preprocess_with_illumination(self):
        config = PreprocessingConfig(
            hough_threshold=10,
            enable_illumination_normalization=True,
            illumination_method="morphology",
        )
        result = preprocess_ecg_image(self.mock_image, config)

        self.assertEqual(result.image.shape, (100, 100, 3))
        self.assertTrue(result.diagnostics.illumination_applied)
        self.assertEqual(result.diagnostics.illumination_method_used, "morphology")

    def test_fallback_behavior(self):
        # pure black image -> no lines
        black_img = np.zeros((100, 100, 3), dtype=np.uint8)
        config = PreprocessingConfig(
            fail_on_missing_lines=False, default_rotation_angle=0.0
        )

        result = preprocess_ecg_image(black_img, config)
        self.assertEqual(result.rotation_angle, 0.0)
        self.assertTrue(result.diagnostics.fallback_used)

    def test_invalid_image_source(self):
        with self.assertRaises(ImageLoadError):
            preprocess_ecg_image("made_up_path_that_doesnt_exist_123.jpg")
