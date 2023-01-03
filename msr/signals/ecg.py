from typing import Dict, List, Type, Union

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
from scipy.signal import find_peaks

from msr.signals.base import (
    BeatSignal,
    MultiChannelPeriodicSignal,
    PeriodicSignal,
    find_intervals_using_hr,
)
from msr.signals.features import get_basic_signal_features
from msr.signals.utils import parse_feats_to_array
from msr.utils import lazy_property

MIN_HR, MAX_HR = 30, 200


def check_ecg_polarity(data):
    try:
        sig = data - np.mean(data)
        pos_peaks, _ = find_peaks(sig)
        neg_peaks, _ = find_peaks(-sig)
        pos_peaks_amps = sig[pos_peaks]
        neg_peaks_amps = abs(sig[neg_peaks])
        pos_amp = np.quantile(pos_peaks_amps, 0.95)
        neg_amp = np.quantile(neg_peaks_amps, 0.95)
        if pos_amp >= neg_amp:
            return 1
        return -1
    except IndexError:
        return 1


class ECGSignal(PeriodicSignal):
    def __init__(self, name, data, fs, start_sec=0):
        polarity = check_ecg_polarity(nk.ecg_clean(data, sampling_rate=fs))
        super().__init__(name, polarity * data, fs, start_sec)
        self.units = "Voltage [V]"
        # self.nk_signals_df, self.nk_info = nk.ecg_process(self.data, sampling_rate=self.fs)
        self.feature_extraction_funcs.update(
            {
                "hrv": self.extract_hrv_features,
            }
        )

    @property
    def cleaned(self):
        return nk.ecg_clean(self.data, sampling_rate=self.fs)

    @property
    def BeatClass(self):
        return ECGBeat

    def find_peaks(self):
        def is_peaks_valid(peaks):
            if len(peaks) == 0:
                return False
            min_n_beats, max_n_beats = MIN_HR / 60 * self.duration, MAX_HR / 60 * self.duration
            return min_n_beats <= len(peaks) <= max_n_beats

        try:
            peaks = nk.ecg_peaks(self.cleaned, sampling_rate=self.fs, method="neurokit")[1]["ECG_R_Peaks"]
            if not is_peaks_valid(peaks):
                raise Exception
        except Exception:  # Error from neurokit2 (IndexError)
            try:
                peaks = nk.ecg_peaks(self.cleaned, sampling_rate=self.fs, method="elgendi2010")[1]["ECG_R_Peaks"]
                if not is_peaks_valid(peaks):
                    raise Exception
            except Exception:  # TODO # both methods raise exceptions
                intervals = find_intervals_using_hr(self)
                peaks = np.array([self.data[start:end].argmax()] + start for start, end in intervals)
        return peaks

    def find_troughs(self):
        peaks = zip(self.peaks[:-1], self.peaks[1:])
        return np.array([self.data[prev_peak : next_peak + 1].argmin() + prev_peak for prev_peak, next_peak in peaks])

    def _get_beats_intervals(self, align_to_peak=True):
        try:
            rpeaks = None  # rpeaks = self.peaks
            qrs_epochs = nk.ecg_segment(self.cleaned, rpeaks=rpeaks, sampling_rate=self.fs, show=False)
            beats_times = []
            intervals = []
            prev_end_idx = qrs_epochs["1"].Index.values[-1]
            for idx, beat in qrs_epochs.items():
                if int(idx) == 1:
                    beats_times.append(beat.index.values)
                else:
                    if not align_to_peak:
                        start_idx = np.where(beat.Index.values == prev_end_idx)[0]
                        start_idx = start_idx[0] if len(start_idx) == 1 else 0
                        beat = beat.iloc[start_idx:]
                    beats_times.append(beat.index.values)
                if beat.Index.values[1] >= self.n_samples or beat.Index.values[0] >= self.n_samples:
                    break
                prev_end_idx = beat.Index.values[-1]
                intervals.append([beat.Index.values[0], beat.Index.values[-1]])

            avg_start_time = np.median([beat[0] for beat in beats_times[1:]])
            first_beat_mask = beats_times[0] > avg_start_time
            beats_times[0] = beats_times[0][first_beat_mask]
            intervals[0][0] = intervals[0][0] + (first_beat_mask == False).sum()
            if intervals[-1][1] >= self.n_samples:
                intervals[-1][1] = self.n_samples - 1
            if intervals[0][0] < 0:
                intervals[0][0] = 0
            intervals = np.array(intervals)
        except ZeroDivisionError:
            print("Setting intervals using hr")
            intervals = find_intervals_using_hr(self)
        return intervals

    def extract_hrv_features(self, return_arr=True, **kwargs):
        peaks = self.peaks
        if len(peaks) == 0:
            hr, ibi_mean, ibi_std, mean_val = np.nan, np.nan, np.nan, np.nan
        else:
            vals = self.data[peaks]
            times = self.time[peaks]
            ibi = np.diff(times)
            ibi_mean = np.mean(ibi)
            ibi_std = np.std(ibi)
            mean_val = np.mean(vals)
            hr = self.duration / len(peaks) * 60
        features = {"hr": hr, "ibi_mean": ibi_mean, "ibi_std": ibi_std, "R_val": mean_val}

        # df = nk.ecg_intervalrelated(self.nk_signals_df, sampling_rate=self.fs)
        # features = df.to_dict(orient="records")[0]
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_critpoints_features(self, return_arr=True, critpoint_names=["p", "q", "r", "s", "t"], **kwargs):
        # TODO: find better way of getting pqst locations (now it is taken from beats which are resampled)
        features = {}
        if hasattr(self, "beats"):
            critpoints_times = {k: [] for k in critpoint_names}
            critpoints_vals = {k: [] for k in critpoint_names}

            for beat in self.beats:
                for name in critpoint_names:
                    critpoint_loc = getattr(beat, f"{name}_loc")
                    critpoints_times[name].append(beat.time[critpoint_loc])
                    critpoints_vals[name].append(beat.data[critpoint_loc])
            for name in critpoint_names:
                times = np.array(critpoints_times[name])
                vals = np.array(critpoints_vals[name])
                times_feats = get_basic_signal_features(times)
                times_feats = {f"{name}_times_{k}": v for k, v in times_feats.items()}
                vals_feats = get_basic_signal_features(vals)
                vals_feats = {f"{name}_vals_{k}": v for k, v in vals_feats.items()}
                features = {**features, **times_feats, **vals_feats}

        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_agg_beat_features(self, return_arr=True, plot=False):
        return self.agg_beat.extract_features(plot=plot, return_arr=return_arr)


class ECGBeat(BeatSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0):
        super().__init__(name, data, fs, start_sec, beat_num)
        self.units = "Voltage [V]"

    @lazy_property
    def r_onset_loc(self):
        try:
            r_onset_loc = self.sder.peaks[self.sder.peaks < self.r_loc][-1]
            if r_onset_loc < self.q_loc:
                r_onset_loc = self.q_loc
            return r_onset_loc
        except Exception:  # TODO
            return self.q_loc

    @lazy_property
    def r_loc(self):
        try:
            peaks, _ = find_peaks(self.data)
            peaks_vals = self.data[peaks]
            r_vals = iter(np.flip(np.sort(peaks_vals)))
            r_loc = peaks[np.where(peaks_vals == next(r_vals))[0][0]]
            while r_loc / self.n_samples >= 0.6:  # find first largest peak which occurs below 60% of total time
                next_largest_peak = np.where(peaks_vals == next(r_vals))[0][0]
                r_loc = peaks[next_largest_peak]
            return r_loc
        except StopIteration:
            # TODO
            return self.data.argmax()

    @lazy_property
    def r_offset_loc(self):
        try:
            r_offset_loc = self.sder.peaks[self.sder.peaks > self.r_loc][0]
            if r_offset_loc > self.s_loc:
                r_offset_loc = self.s_loc
            return r_offset_loc
        except IndexError:  # TODO
            return self.s_loc

    @lazy_property
    def p_onset_loc(self):
        try:
            p_onset_loc = self.sder.peaks[self.sder.peaks < self.p_loc][-1]
        except Exception:  # TODO
            p_onset_loc = 0
        return p_onset_loc

    @lazy_property
    def p_loc(self):
        try:
            zero_to_r_data = self.data[: self.r_loc]
            peaks, _ = find_peaks(zero_to_r_data)
            biggest_peak_idx = np.array([zero_to_r_data[peak] for peak in peaks]).argmax()
            return peaks[biggest_peak_idx]  # zero_to_r_data.argmax()
        except Exception:  # TODO
            return int(self.r_loc / 4)

    @lazy_property
    def p_offset_loc(self):
        try:
            p_offset_loc = self.sder.peaks[self.sder.peaks > self.p_loc][0]
            if p_offset_loc > self.q_loc:
                p_offset_loc = self.q_loc
            return p_offset_loc
        except IndexError:  # TODO
            return self.q_loc

    @lazy_property
    def q_loc(self):
        try:
            p_to_r_data = self.data[self.p_loc : self.r_loc]
            return p_to_r_data.argmin() + self.p_loc
        except ValueError:  # TODO
            return self.r_loc - (self.r_loc - self.p_loc) // 3

    @lazy_property
    def s_loc(self):
        try:
            r_to_end_data = self.data[self.r_loc :]
            troughs, _ = find_peaks(-r_to_end_data)
            return troughs[0] + self.r_loc
        except IndexError:  # TODO
            return self.r_loc + (self.n_samples - self.r_loc) // 4

    @lazy_property
    def t_onset_loc(self):
        try:
            t_onset_loc = self.sder.peaks[self.sder.peaks < self.t_loc][-1]
            if t_onset_loc < self.s_loc:
                t_onset_loc = self.s_loc
        except Exception:  # TODO
            t_onset_loc = self.s_loc
        return t_onset_loc

    @lazy_property
    def t_loc(self):
        try:
            s_to_end_data = self.data[self.s_loc :]
            peaks, _ = find_peaks(s_to_end_data)
            biggest_peak_idx = np.array([s_to_end_data[peak] for peak in peaks]).argmax()
            return peaks[biggest_peak_idx] + self.s_loc  # zero_to_r_data.argmax()
        except Exception:  # TODO
            return self.s_loc + (self.n_samples - self.s_loc) // 2

    @lazy_property
    def t_offset_loc(self):
        try:
            t_offset_loc = self.sder.peaks[self.sder.peaks > self.t_loc][0]
            return t_offset_loc
        except Exception:  # TODO
            return self.n_samples - 1

    @lazy_property
    def crit_points(self):
        return {
            "p_onset": self.p_onset_loc,
            "p": self.p_loc,
            "p_offset": self.p_offset_loc,
            "q": self.q_loc,
            "r_onset": self.r_onset_loc,
            "r": self.r_loc,
            "r_offset": self.r_offset_loc,
            "s": self.s_loc,
            "t_onset": self.t_onset_loc,
            "t": self.t_loc,
            "t_offset": self.t_offset_loc,
        }

    @lazy_property
    def energy_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        return [
            {"name": "ZeroPQ_E", "start": self.p_onset_loc, "end": self.p_offset_loc},
            {"name": "QRS_E", "start": self.r_onset_loc, "end": self.r_offset_loc},
            {"name": "STEnd_E", "start": self.t_onset_loc, "end": self.t_offset_loc},
        ]

    @lazy_property
    def area_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        return [
            {"name": "P_A", "start": self.p_onset_loc, "end": self.p_offset_loc},
            {"name": "QRS_A", "start": self.r_onset_loc, "end": self.r_offset_loc},
            {"name": "T_A", "start": self.t_onset_loc, "end": self.t_offset_loc},
        ]

    @lazy_property
    def slope_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        return [
            {"name": "P_onset_slope", "start": self.p_onset_loc, "end": self.p_loc},
            {"name": "P_offset_slope", "start": self.p_loc, "end": self.p_offset_loc},
            {"name": "R_onset_slope", "start": self.r_onset_loc, "end": self.r_loc},
            {"name": "R_offset_slope", "start": self.r_loc, "end": self.r_offset_loc},
            {"name": "T_onset_slope", "start": self.t_onset_loc, "end": self.t_loc},
            {"name": "T_offset_slope", "start": self.t_loc, "end": self.t_offset_loc},
        ]

    @lazy_property
    def intervals_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        return [
            {"name": "PQ_segment", "start": self.p_offset_loc, "end": self.q_loc},
            {"name": "ST_segment", "start": self.s_loc, "end": self.t_onset_loc},
            {"name": "PR_interval", "start": self.p_onset_loc, "end": self.r_offset_loc},
            {"name": "QRS_interval", "start": self.q_loc, "end": self.s_loc},
            {"name": "QT_interval", "start": self.q_loc, "end": self.t_offset_loc},
        ]


class MultiChannelECGSignal(MultiChannelPeriodicSignal):
    def plot_beats_segmentation(self, valid=True, invalid=True, use_raw=False, color_validity=False, **kwargs):
        fig, axes = plt.subplots(
            self.n_signals, 2, figsize=(24, 1.6 * self.n_signals), sharex="col", gridspec_kw={"width_ratios": [9, 2]}
        )
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot_beats_segmentation(
                valid=valid, invalid=invalid, use_raw=use_raw, color_validity=color_validity, axes=ax
            )
            sig.agg_beat.plot_crit_points(points=["p", "q", "r", "s", "t"], ax=ax[1])
            ax[0].set_title(sig_name)
            ax[1].set_title("")
        plt.tight_layout()
        return fig
