import unittest
import numpy as np

from preprocessing.enhance import (
    to_agnostic_grayscale,
    standardize_resolution,
    isolate_grid_lines,
)


class TestEnhancementModule(unittest.TestCase):
    def setUp(self):
        # Create an RGB dummy block with synthetic gridlines
        self.mock_rgb = np.full((1000, 500, 3), 255, dtype=np.uint8)

        # Draw some pink lines mimicking an ECG grid (L-channel conversion will normalize these natively)
        # horizontal
        for y in range(0, 1000, 20):
            self.mock_rgb[y, :, :] = [255, 192, 203]  # Pink in BGR/RGB approximation
        # vertical
        for x in range(0, 500, 20):
            self.mock_rgb[:, x, :] = [255, 192, 203]

    def test_agnostic_grayscale(self):
        gray = to_agnostic_grayscale(self.mock_rgb)

        # Verify it drops to a 1D Channel
        self.assertEqual(len(gray.shape), 2)
        self.assertEqual(gray.shape, (1000, 500))

        # If passed an already 1D array, verify it just returns itself safely
        gray2 = to_agnostic_grayscale(gray)
        self.assertEqual(gray2.shape, (1000, 500))

    def test_standardize_resolution(self):
        # We explicitly set target to 200. Max edge is currently 1000. Scale factor = 200/1000 = 0.2
        # Target new dimensions should be: (1000*0.2, 500*0.2) = (200, 100)
        scaled = standardize_resolution(self.mock_rgb, target_long_edge=200)

        self.assertEqual(scaled.shape[:2], (200, 100))

        # What if it's already the target size? It should pass through unscathed
        scaled_again = standardize_resolution(scaled, target_long_edge=200)
        np.testing.assert_array_equal(scaled, scaled_again)

    def test_isolate_grid_lines(self):
        gray = to_agnostic_grayscale(self.mock_rgb)

        # Let's add some noisy text or 'signal' blobs that aren't perfectly straight orthongal lines
        gray[50:100, 50:100] = 50

        # Isolate using a strict kernel lengths
        isolated = isolate_grid_lines(gray, morphological_length=200)

        # The morphological grids will preserve the straight 500px line traces
        # But delete the 50x50 noise blob

        self.assertEqual(isolated.shape, (1000, 500))
        # Given it is pure contrast matrix, test standard distribution assumptions
        # (mostly pure white background, dark isolated lines)
        self.assertTrue(isolated.mean() > 200)
