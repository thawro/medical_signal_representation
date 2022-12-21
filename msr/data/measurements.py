import numpy as np

from msr.signals.base import MultiChannelPeriodicSignal, MultiChannelSignal
from msr.signals.ecg import ECGSignal, MultiChannelECGSignal
from msr.signals.eeg import EEGSignal
from msr.signals.eog import EOGSignal
from msr.signals.ppg import PPGSignal
from msr.signals.utils import parse_feats_to_array


class PtbXLMeasurement(MultiChannelECGSignal):
    def __init__(self, *ecg_leads: np.ndarray, fs: float = 100):
        signals = {f"ecg_{i+1}": ECGSignal(f"ecg_{i+1}", ecg_lead, fs) for i, ecg_lead in enumerate(ecg_leads)}
        super().__init__(signals)
        self.feature_extraction_funcs.update({"some_features": self.extract_some_features})  # TODO

    def extract_some_features(self, return_arr=True, plot=False):
        # TODO
        features = {"some_feature": 0}
        if return_arr:
            return parse_feats_to_array(features)
        return features


class SleepEDFMeasurement(MultiChannelSignal):
    def __init__(self, eeg_0: np.ndarray, eeg_1: np.ndarray, eog: np.ndarray, fs: float = 100):
        signals = {
            "eeg_0": EEGSignal("eeg_0", eeg_0, fs),  # EEG Fpz-Cz
            "eeg_1": EEGSignal("eeg_1", eeg_1, fs),  # EEG Pz-Oz
            "eog": EOGSignal("eog", eog, fs),  # EOG horizontal
        }
        super().__init__(signals)
        self.feature_extraction_funcs.update({"some_features": self.extract_some_features})  # TODO

    def extract_some_features(self, return_arr=True, plot=False):
        # TODO
        features = {"some_feature": 0}
        if return_arr:
            return parse_feats_to_array(features)
        return features


class MimicMeasurement(MultiChannelPeriodicSignal):
    def __init__(self, ppg: np.ndarray, ecg: np.ndarray, fs: float = 125):
        signals = {
            "ppg": PPGSignal("ppg", ppg, fs),
            "ecg": ECGSignal("ecg", ecg, fs),
        }
        super().__init__(signals)
        self.feature_extraction_funcs.update({"ptt_features": self.extract_ptt_features})

    def extract_ptt_features(self, return_arr=True, plot=False):
        features = {"ptt": 0}
        if return_arr:
            return parse_feats_to_array(features)
        return features
