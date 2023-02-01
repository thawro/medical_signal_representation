import matplotlib.pyplot as plt
import numpy as np

from msr.signals.base import MultiChannelPeriodicSignal, MultiChannelSignal
from msr.signals.ecg import ECGSignal
from msr.signals.eeg import EEGSignal
from msr.signals.eog import EOGSignal
from msr.signals.features import get_basic_signal_features
from msr.signals.ppg import PPGSignal
from msr.signals.utils import parse_feats_to_array


class PtbXLMeasurement(MultiChannelPeriodicSignal):
    def __init__(self, *ecg_leads: np.ndarray, fs: float = 100):
        signals = {f"ecg_{i+1}": ECGSignal(f"ecg_{i+1}", ecg_lead, fs) for i, ecg_lead in enumerate(ecg_leads)}
        super().__init__(signals)
        # self.feature_extraction_funcs.update({"some_features": self.extract_some_features})  # TODO

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
        # self.feature_extraction_funcs.update({"some_features": self.extract_some_features})  # TODO

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
        self.feature_extraction_funcs.update({"ptt": self.extract_ptt_features})

    def get_ptt(self):
        ecg = self.signals["ecg"]
        ppg = self.signals["ppg"]

        ecg_intervals = (np.array([[beat.start_sec, beat.end_sec] for beat in ecg.beats]) * ecg.fs).astype(int)
        ecg_peaks = np.array([ecg.cleaned[s:e].argmax() + s for s, e in ecg_intervals])
        ppg_peaks = np.array([ppg.cleaned[s:e].argmax() + s for s, e in ecg_intervals])

        ptt = ecg_peaks - ppg_peaks
        n_positive_ptt = (ptt > 0).sum()
        positive_ptt_ratio = n_positive_ptt / len(ecg_peaks)
        if positive_ptt_ratio < 0.5:
            ppg_peaks, ecg_peaks = ppg_peaks[:-1], ecg_peaks[1:]
            ptt = ecg_peaks - ppg_peaks

        ptt_locs = ecg_peaks.copy()
        positive_ptt_mask = ptt > 0

        ptt_time = ptt_locs[positive_ptt_mask] / ecg.fs
        ptt = ptt[positive_ptt_mask] / ecg.fs
        return ptt_time, ptt

    def extract_ptt_features(self, return_arr=True, plot=False, ax=None):
        ptt_time, ptt = self.get_ptt()
        features = get_basic_signal_features(ptt)
        return_fig = False
        if plot:
            if ax is None:
                return_fig = True
                fig, ax = plt.subplots(1, 1, figsize=self.signals["ecg"].fig_params["fig_size"])
            ax.plot(ptt_time, ptt)

        if return_arr:
            return parse_feats_to_array(features)

        if return_fig:
            return features, fig
        return features
