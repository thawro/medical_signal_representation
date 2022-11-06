from collections import OrderedDict

import numpy as np

from msr.signals.ecg import ECGSignal, MultiChannelECGSignal
from msr.signals.utils import parse_feats_to_array


class PtbXLMeasurement(MultiChannelECGSignal):
    def __init__(self, ecg_leads: np.ndarray, fs: float = 100):
        signals = OrderedDict(
            {f"ecg_{i+1}": ECGSignal(f"ecg_{i+1}", ecg_lead, fs) for i, ecg_lead in enumerate(ecg_leads)}
        )
        super().__init__(signals)
        self.feature_extraction_funcs.update({"some_features": self.extract_some_features})  # TODO

    def extract_some_features(self, return_arr=True, plot=False):
        # TODO
        features = OrderedDict({"some_feature": 0})
        if return_arr:
            return parse_feats_to_array(features)
        return features
