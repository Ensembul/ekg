class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""

    pass


class ImageLoadError(PreprocessingError):
    """Failed to load or decode image from path/data."""

    pass


class RotationEstimationError(PreprocessingError):
    """Failed to robustly estimate a rotation angle."""

    pass


class IlluminationError(PreprocessingError):
    """Failed to apply illumination normalization."""

    pass


class IlluminationConfigError(IlluminationError):
    """Invalid illumination configuration provided."""

    pass
