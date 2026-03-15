import sys
import unittest


class TestNoTorchDependency(unittest.TestCase):
    def test_torch_not_imported(self):
        """
        Guard establishing that loading the preprocessing pipeline
        does not invoke or pull any PyTorch frameworks.
        """
        # Ensure it's not already in sys.modules globally causing a false positive
        if "torch" in sys.modules:
            del sys.modules["torch"]
        if "torchvision" in sys.modules:
            del sys.modules["torchvision"]

        self.assertNotIn(
            "torch", sys.modules, "Found unexpected torch import in preprocessing."
        )
        self.assertNotIn(
            "torchvision",
            sys.modules,
            "Found unexpected torchvision import in preprocessing.",
        )

    def test_no_legacy_imports(self):
        """
        Guard establishing that no legacy modules are implicitly imported
        """
        self.assertNotIn("src.run.digitize", sys.modules)
        self.assertNotIn(
            "config", sys.modules
        )  # meaning the top-level ECG-digitiser config
