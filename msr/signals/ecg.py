from collections import OrderedDict

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import seaborn as sns
from scipy.signal import find_peaks

from msr.signals.base import BeatSignal, MultiChannelPeriodicSignal, PeriodicSignal
from msr.signals.utils import (
    calculate_area,
    calculate_energy,
    calculate_slope,
    parse_feats_to_array,
)
from msr.utils import lazy_property

CHANNELS_POLARITY = [1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1]  # some channels are with oposite polarity


def check_ecg_polarity(data):
    try:
        sig = data - data.mean()
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


def find_intervals_using_hr(sig):
    freqs, amps = sig.psd(window_size=10, plot=False)
    most_dominant_freq = freqs[amps.argmax()]  # HR
    T_in_samples = int(1 / most_dominant_freq * sig.fs)
    x = sig.cleaned
    x_minmax = -(x - x.min()) / (x.max() - x.min()) + 1
    troughs, _ = find_peaks(x_minmax, height=0.9)
    intervals = [(troughs[0], troughs[0] + T_in_samples)]
    while intervals[-1][1] + T_in_samples < sig.n_samples:
        prev_end = intervals[-1][1]
        intervals.append((prev_end, prev_end + T_in_samples))
    return intervals


def create_multichannel_ecg(data, fs):
    signals = OrderedDict({i + 1: ECGSignal(f"ECG_{i+1}", channel_data, fs) for i, channel_data in enumerate(data)})
    return MultiChannelECGSignal(signals)


class ECGSignal(PeriodicSignal):
    def __init__(self, name, data, fs, start_sec=0):
        polarity = check_ecg_polarity(nk.ecg_clean(data, sampling_rate=fs))
        super().__init__(name, polarity * data, fs, start_sec)
        self.cleaned = nk.ecg_clean(self.data, sampling_rate=self.fs)
        self.feature_extraction_funcs.update(
            {
                "hrv_features": self.extract_hrv_features,
            }
        )

    @lazy_property
    def rpeaks(self):
        return nk.ecg_peaks(self.cleaned, sampling_rate=self.fs)[1]["ECG_R_Peaks"]

    def set_beats(
        self,
        intervals=None,
        align_to_r=True,
        resample=True,
        n_samples=100,
        plot=False,
        use_raw=False,
    ):
        if intervals is None:
            try:
                qrs_epochs = nk.ecg_segment(self.cleaned, rpeaks=None, sampling_rate=self.fs, show=False)
                beats_times = []
                intervals = []
                prev_end_idx = qrs_epochs["1"].Index.values[-1]
                for idx, beat in qrs_epochs.items():
                    if int(idx) == 1:
                        beats_times.append(beat.index.values)
                    else:
                        if not align_to_r:
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
            except ZeroDivisionError:
                intervals = find_intervals_using_hr(self)
        intervals = np.array(intervals)
        if not resample:
            intervals[:, 1] = intervals[:, 0] + n_samples
        self.beats_intervals = intervals
        self.valid_beats_mask = len(intervals) * [True]
        data_source = self.data if use_raw else self.cleaned
        beats = []
        for i, interval in enumerate(intervals):
            start_sec = interval[0] / self.fs
            beat_data = data_source[interval[0] : interval[1] + 1]
            beat = ECGBeat(name=f"beat {i}", data=beat_data, fs=self.fs, start_sec=start_sec, beat_num=i)
            if resample:
                beat = beat.resample_with_interpolation(n_samples=100, kind="pchip")
            beats.append(beat)
        self.beats = np.array(beats)
        if plot:
            self.plot_beats()

    def extract_hrv_features(self, return_arr=True, **kwargs):
        if self.rpeaks is None:
            self.find_rpeaks()
        r_peaks = self.rpeaks
        r_vals = self.data[r_peaks]
        r_times = self.time[r_peaks]
        ibi = np.diff(r_times)
        features = OrderedDict({"ibi_mean": np.mean(ibi), "ibi_std": np.std(ibi), "R_val": np.mean(r_vals)})
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_agg_beat_features(self, return_arr=True, plot=False):
        return self.agg_beat.extract_features(plot=plot, return_arr=return_arr)


class ECGBeat(BeatSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0):
        super().__init__(name, data, fs, start_sec, beat_num)
        self.feature_extraction_funcs.update(
            {
                "area_features": self.extract_area_features,
                "energy_features": self.extract_energy_features,
                "slope_features": self.extract_slope_features,
                "energy_features": self.extract_energy_features,
            }
        )

    @lazy_property
    def fder(self):
        return self.get_derivative(1)

    @lazy_property
    def sder(self):
        return self.get_derivative(2)

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

    def extract_area_features(self, return_arr=True, method="trapz", plot=False, ax=None):
        params = [
            {"name": "P_A", "start": self.p_onset_loc, "end": self.p_offset_loc},
            {"name": "QRS_A", "start": self.r_onset_loc, "end": self.r_offset_loc},
            {"name": "T_A", "start": self.t_onset_loc, "end": self.t_offset_loc},
        ]
        features = OrderedDict(
            {p["name"]: calculate_area(self.data, self.fs, p["start"], p["end"], method=method) for p in params}
        )
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(14, 6))
            palette = sns.color_palette("pastel", len(params))

            ax.plot(self.time, self.data, lw=3)
            self.plot_crit_points(ax=ax)
            areas = []
            for i, p in enumerate(params):
                offset = min([self.data[p["start"]], self.data[p["end"]]])
                mask = (np.arange(self.n_samples) >= p["start"]) & (np.arange(self.n_samples) <= p["end"])
                a = ax.fill_between(
                    self.time[mask],
                    self.data[mask],
                    offset,
                    alpha=0.4,
                    color=palette[i],
                    edgecolor="black",
                    lw=1,
                    label=p["name"],
                )
                areas.append(a)
            leg = ax.legend(handles=areas, fontsize=14)
            ax.add_artist(leg)
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_energy_features(self, return_arr=True, **kwargs):
        params = [
            {"name": "ZeroPQ_E", "start": self.p_onset_loc, "end": self.p_offset_loc},
            {"name": "QRS_E", "start": self.r_onset_loc, "end": self.r_offset_loc},
            {"name": "STEnd_E", "start": self.t_onset_loc, "end": self.t_offset_loc},
        ]

        features = OrderedDict({p["name"]: calculate_energy(self.data, p["start"], p["end"]) for p in params})
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_slope_features(self, return_arr=True, plot=False, ax=None):
        params = [
            {"name": "P_onset_slope", "start": self.p_onset_loc, "end": self.p_loc},
            {"name": "P_offset_slope", "start": self.p_loc, "end": self.p_offset_loc},
            {"name": "R_onset_slope", "start": self.r_onset_loc, "end": self.r_loc},
            {"name": "R_offset_slope", "start": self.r_loc, "end": self.r_offset_loc},
            {"name": "T_onset_slope", "start": self.t_onset_loc, "end": self.t_loc},
            {"name": "T_offset_slope", "start": self.t_loc, "end": self.t_offset_loc},
        ]

        features = OrderedDict({p["name"]: calculate_slope(self.time, self.data, p["start"], p["end"]) for p in params})
        if return_arr:
            return parse_feats_to_array(features)
        return features


class MultiChannelECGSignal(MultiChannelPeriodicSignal):
    def plot_beats_segmentation(self, use_raw=False, **kwargs):
        fig, axes = plt.subplots(
            self.n_signals, 2, figsize=(24, 1.3 * self.n_signals), sharex="col", gridspec_kw={"width_ratios": [9, 2]}
        )
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot_beats_segmentation(use_raw=use_raw, axes=ax)
            sig.agg_beat.plot_crit_points(points=["p", "q", "r", "s", "t"], ax=ax[1])
            ax[1].set_title("")
        plt.tight_layout()

    def set_beats(self, source_channel=2, return_arr=True, **kwargs):
        """Return beats from all channels.

        If `source_channel` is specified, it will be used as source of beat intervals
        for other channels. If `ZeroDivisionError` occurs (from `neurokit2`) for specified source_channel,
        other channels are tested.
        If `source_channel` is `None`, beats are extracter for every channel separately. In that case, there
        is no guarantee, that number of beats and their alignment is preserved across all channels.

        Args:
            source_channel (int, optional): Channel used as source of beat intervals. Defaults to `2`.
            return_arr (bool, optional): Whether to return beats data as `np.ndarray` or :class:`ECGBeat` objects.
                Defaults to `True`.

        Returns:
            _type_: Beats as `np.ndarray` or as :class:`ECGBeat` objects.
        """
        if source_channel is not None:
            source_channels = list(self.signals.keys())
            source_channels.remove(source_channel)
            source_channels.reverse()
            found_good_channel = False
            new_source_channel = source_channel
            while not found_good_channel or len(source_channels) == 0:
                try:
                    self.signals[new_source_channel].set_beats(**kwargs)
                    found_good_channel = True
                    if new_source_channel != source_channel:
                        print(f"Original source channel was corrupted. Found new source channel: {new_source_channel}")
                except ZeroDivisionError:
                    new_source_channel = source_channels.pop()
            if not found_good_channel:
                new_source_channel = source_channel
                self.signals[new_source_channel].set_beats(**kwargs)  # use Heart Rate (most dominant frequency)
            source_channel_intervals = self.signals[new_source_channel].beats_intervals
        else:
            source_channel_intervals = None
        for _, signal in self.signals.items():
            signal.set_beats(source_channel_intervals, **kwargs)
        all_channels_beats = [signal.beats for _, signal in self.signals.items()]
        self.beats = np.array(all_channels_beats)
        self.beats_times = np.array([beat.time for beat in all_channels_beats[0]])


def create_multichannel_ecg(data, fs):
    signals = OrderedDict(
        {f"ecg_{i+1}": ECGSignal(f"ecg_{i+1}", channel_data, fs) for i, channel_data in enumerate(data)}
    )
    return MultiChannelECGSignal(signals)
