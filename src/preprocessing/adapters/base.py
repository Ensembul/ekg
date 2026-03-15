class SegmenterAdapter:
    """
    Base protocol definition for plugging the standalone preprocessor
    results into a discrete segmentation graph/framework (like nnUNet).
    """

    def prepare_input(self, preprocessed_ecg):
        raise NotImplementedError

    def predict(self, preprocessed_ecg):
        raise NotImplementedError
