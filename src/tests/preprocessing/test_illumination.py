import unittest
import numpy as np

from preprocessing.illumination import apply_illumination_normalization
from preprocessing.models import PreprocessingConfig
from preprocessing.errors import IlluminationConfigError


class TestIlluminationNormalization(unittest.TestCase):
    def setUp(self):
        # Create a mock grayscale base image with a broad dark shadow and thin traces
        self.mock_image = np.full((100, 100, 3), 200, dtype=np.uint8)
        # Create a broad vertical "crease" shadow (larger than kernel 25)
        self.mock_image[:, 20:80, :] = 100
        # Thin dark line (ECG trace) in bright area, value 20
        self.mock_image[50:53, 5:15, :] = 20
        # Thin dark line (ECG trace) inside the shadow, value 20
        self.mock_image[50:53, 40:60, :] = 20

    def test_illumination_disabled_by_default(self):
        config = PreprocessingConfig()  # default False
        result = apply_illumination_normalization(self.mock_image, config)

        # Should return exactly the exact same memory object / unchanged array
        np.testing.assert_array_equal(result, self.mock_image)

    def test_morphology_output_bounds(self):
        config = PreprocessingConfig(
            enable_illumination_normalization=True, illumination_method="morphology"
        )
        result = apply_illumination_normalization(self.mock_image, config)

        self.assertEqual(result.shape, (100, 100, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(result.max() <= 255)
        self.assertTrue(result.min() >= 0)

        # Verify background became uniformly white (~255)
        self.assertTrue(result[10, 10, 0] > 240)
        # Verify the broad shadow was removed to near white
        self.assertTrue(result[10, 50, 0] > 240)
        # Verify trace in bright area is still dark
        self.assertTrue(result[51, 10, 0] < 100)
        # Verify trace in shadow area is still dark
        self.assertTrue(result[51, 50, 0] < 100)

    def test_clahe_output_bounds(self):
        config = PreprocessingConfig(
            enable_illumination_normalization=True, illumination_method="clahe"
        )
        result = apply_illumination_normalization(self.mock_image, config)

        self.assertEqual(result.shape, (100, 100, 3))
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(result.max() <= 255)
        self.assertTrue(result.min() >= 0)

    def test_invalid_method_raises(self):
        config = PreprocessingConfig(
            enable_illumination_normalization=True,
            illumination_method="nonexistent_method",
        )
        with self.assertRaises(IlluminationConfigError):
            apply_illumination_normalization(self.mock_image, config)

    def test_deterministic_output(self):
        config = PreprocessingConfig(
            enable_illumination_normalization=True, illumination_method="clahe"
        )
        result_1 = apply_illumination_normalization(self.mock_image, config)
        result_2 = apply_illumination_normalization(self.mock_image, config)

        np.testing.assert_array_equal(result_1, result_2)
