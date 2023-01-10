from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import ChainMap
from typing import Dict, List, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from nptyping import Float, Int, NDArray, Shape
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import (
    cheby2,
    detrend,
    filtfilt,
    find_peaks,
    firwin,
    iirnotch,
    resample,
    savgol_filter,
    sosfiltfilt,
    welch,
)
from sorcery import dict_of
from statsmodels.tsa.seasonal import STL

from msr.signals.features import (
    extract_dwt_features,
    get_basic_signal_features,
    get_cwt_img,
)
from msr.signals.utils import (
    BEAT_FIG_PARAMS,
    SIGNAL_FIG_PARAMS,
    calculate_area,
    calculate_energy,
    calculate_slope,
    find_closest_element,
    get_valid_beats_mask,
    get_windows,
    interpolate_to_new_time,
    parse_feats_to_array,
    parse_nested_feats,
    z_score,
)
from msr.utils import create_new_obj, lazy_property

log = logging.getLogger(__name__)

MIN_HR, MAX_HR = 30, 200


def find_intervals_using_most_dominant_freq(signal: Type[PeriodicSignal]):
    T_in_samples = int(1 / signal.most_dominant_freq * signal.fs)
    x = signal.cleaned
    neg_x_minmax = -(x - x.min()) / (x.max() - x.min()) + 1
    troughs, _ = find_peaks(neg_x_minmax, height=0.5)
    # troughs = signal.troughs
    intervals = [(troughs[0], find_closest_element(troughs, troughs[0] + T_in_samples)[1])]
    while intervals[-1][1] + T_in_samples < signal.n_samples:
        prev_end = intervals[-1][1]
        next_end = find_closest_element(troughs, prev_end + T_in_samples)[1]
        intervals.append((prev_end, next_end))
    return np.array(intervals)


class BaseSignal:
    def __init__(self, name: str, data: NDArray[Shape["N"], Float], fs: float, start_sec: float = 0):
        self.name = name
        self.data = data
        self.fs = fs
        self.min = self.cleaned.min()
        self.max = self.cleaned.max()
        self.range = self.max - self.min

        self.units = "Values [a.u.]"
        self.start_sec = start_sec
        self.n_samples = len(data)
        self.duration = self.n_samples / fs
        self.time = np.arange(0, self.n_samples) / fs
        self.end_sec = self.start_sec + self.time[-1]
        self.fig_params = SIGNAL_FIG_PARAMS
        self.feature_extraction_funcs = {
            "basic": self.extract_basic_features,
        }

    @property
    def cleaned(self):
        return np.copy(self.data)

    def find_peaks(self):
        return find_peaks(self.cleaned)[0]

    def find_troughs(self):
        peaks = zip(self.peaks[:-1], self.peaks[1:])
        return np.array([self.cleaned[prev_peak:next_peak].argmin() + prev_peak for prev_peak, next_peak in peaks])

    @lazy_property
    def troughs(self):
        return self.find_troughs()

    @lazy_property
    def peaks(self):
        return self.find_peaks()

    def interpolate(self, fs, kind="cubic") -> Type[BaseSignal]:
        if kind == "pchip":
            interp_fn = PchipInterpolator(self.time, self.data)
        else:
            interp_fn = interp1d(self.time, self.data, kind=kind, fill_value="extrapolate")
        n_samples = int(self.n_samples * fs / self.fs)
        new_time = np.arange(0, n_samples) / fs
        new_time = new_time[(new_time >= self.time[0]) & (new_time <= self.time[-1])]
        interpolated = interp_fn(new_time)
        new_name = self.name + f"_interp({fs:.2f}Hz)"
        return create_new_obj(self, name=new_name, data=interpolated, fs=fs)

    def resample(self, n_samples) -> Type[BaseSignal]:
        new_data = resample(self.data, n_samples)
        fs_scaler = len(new_data) / self.n_samples
        new_fs = self.fs * fs_scaler
        new_name = self.name + f"_resampled({n_samples})"
        return create_new_obj(self, name=new_name, data=new_data, fs=new_fs)

    def resample_with_interpolation(self, n_samples, kind="pchip") -> Type[BaseSignal]:
        fs_scaler = n_samples / self.n_samples
        new_fs = self.fs * fs_scaler
        if kind == "pchip":
            interp_fn = PchipInterpolator(self.time, self.data)
        else:
            interp_fn = interp1d(self.time, self.data, kind=kind, fill_value="extrapolate")
        new_time = np.arange(0, n_samples) / new_fs
        interpolated = interp_fn(new_time)
        new_name = self.name + f"_interp({n_samples})"
        return create_new_obj(self, name=new_name, data=interpolated, fs=new_fs)

    def z_score(self) -> Type[BaseSignal]:
        new_data = z_score(self.data)
        new_name = self.name + "_z_scored"
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

    def min_max(self) -> Type[BaseSignal]:
        arr = self.data
        new_data = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        new_name = self.name + "_min_max"
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

    def get_slice(self, start_time, end_time) -> Type[BaseSignal]:
        mask = (self.time >= start_time) & (self.time < end_time)
        new_name = self.name + f"_slice({start_time}-{end_time})"
        new_data = self.data[mask]
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

    def is_in_time_bounds(self, start_s: float, end_s: float) -> bool:
        return self.start_sec >= start_s and self.end_sec < end_s

    def get_derivative(self, deriv=1, window_length=None, polyorder=4) -> BaseSignal:
        if window_length is None:
            window_length = int(np.ceil(15 / 100 * self.fs))
        if window_length % 2 == 0:
            window_length = window_length + 1
        if window_length <= polyorder:
            window_length = polyorder + 1
        new_name = self.name + f"_derivative({deriv})"
        new_data = savgol_filter(self.data, window_length, polyorder, deriv, delta=1 / self.fs)
        return BaseSignal(name=new_name, data=new_data, fs=self.fs, start_sec=self.start_sec)

    @lazy_property
    def fder(self) -> BaseSignal:
        return self.get_derivative(1)

    @lazy_property
    def sder(self) -> BaseSignal:
        return self.get_derivative(2)

    def detrend(self) -> Type[BaseSignal]:
        detrended = detrend(self.data)
        new_name = self.name + f"_detrended"
        return create_new_obj(self, name=new_name, data=detrended, fs=self.fs)

    def get_cheby2_filtered(self, cutoff, btype="lowpass", order=4, rs=10) -> Type[BaseSignal]:
        sos = cheby2(order, rs, cutoff, btype, output="sos", fs=self.fs)
        filtered = sosfiltfilt(sos, self.data)
        new_name = self.name + f"_{btype}_cheby2_filt({np.round(cutoff, 2)})"
        return create_new_obj(self, name=new_name, data=filtered, fs=self.fs)

    def get_fir_filtered(self, cutoff, btype="lowpass", numtaps=10) -> Type[BaseSignal]:
        firf = firwin(numtaps, cutoff, pass_zero=btype, fs=self.fs)
        filtered = filtfilt(firf, 1, self.data)
        new_name = self.name + f"{btype}_fir_filt({np.round(cutoff, 2)})"
        return create_new_obj(self, name=new_name, data=filtered, fs=self.fs)

    def get_notch_filtered(self, cutoff, Q=10) -> Type[BaseSignal]:
        b, a = iirnotch(cutoff, Q=Q, fs=self.fs)
        filtered = filtfilt(b, a, self.data)
        new_name = self.name + f"_notch_filt({np.round(cutoff, 2)})"
        return create_new_obj(self, name=new_name, data=filtered, fs=self.fs)

    def seasonal_decompose(self, plot=False):
        res = STL(self.data, period=self.fs).fit()
        if plot:
            res.plot()
        return res

    def fft(self, plot=False, ax=None) -> Tuple[NDArray[Shape["N"], Float], NDArray[Shape["N"], Float]]:
        amps = rfft(self.cleaned)
        freqs = rfftfreq(self.n_samples, 1 / self.fs)
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(freqs[1:], np.abs(amps[1:]))
        return freqs, amps

    def psd(
        self, fmin=0, fmax=10, plot=False, ax=None, normalize=True
    ) -> Tuple[NDArray[Shape["N"], Float], NDArray[Shape["N"], Float]]:
        freqs, pxx = welch(self.cleaned, self.fs)
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs, pxx = freqs[mask], pxx[mask]
        if normalize:
            pxx /= sum(pxx)
        return_fig = False
        if plot:
            if ax is None:
                return_fig = True
                fig, ax = plt.subplots(figsize=BEAT_FIG_PARAMS["fig_size"])

            ax.semilogy(freqs, pxx, lw=2)
            ax.set_xlabel("Frequency [Hz]", fontsize=BEAT_FIG_PARAMS["label_size"])
            ax.set_ylabel("PSD", fontsize=BEAT_FIG_PARAMS["label_size"])
            ax.tick_params(axis="both", which="major", labelsize=BEAT_FIG_PARAMS["tick_size"])
        if return_fig:
            return freqs, pxx, fig
        return freqs, pxx

    def spectrogram(
        self, NFFT=256, noverlap=128, ax=None
    ) -> Tuple[NDArray[Shape["N"], Float], NDArray[Shape["N"], Float], NDArray[Shape["N"], Float]]:
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 8))
        spectrum, freqs, t, img = ax.specgram(self.cleaned, Fs=self.fs, NFFT=NFFT, noverlap=noverlap, cmap="viridis")
        cbar = plt.gcf().colorbar(img, ax=ax)
        cbar.ax.set_title("Intensity", fontsize=self.fig_params["title_size"])
        ax.set_ylabel("Frequency [Hz]", fontsize=self.fig_params["label_size"])
        ax.set_xlabel("Time [s]", fontsize=self.fig_params["label_size"])
        ax.tick_params(axis="both", which="major", labelsize=self.fig_params["tick_size"])
        return spectrum, freqs, t

    def get_whole_signal_waveform(self) -> NDArray[Shape["N"], Float]:
        return self.cleaned

    def extract_basic_features(self, return_arr=True, **kwargs) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = get_basic_signal_features(self.cleaned)
        if return_arr:
            return np.array(list(features.values()))
        return features

    def extract_features(self, return_arr=True, plot=False) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = {name: func(return_arr=False, plot=plot) for name, func in self.feature_extraction_funcs.items()}
        features = parse_nested_feats(features)
        self.feature_names = list(features.keys())
        if return_arr:
            return np.array(list(features.values()))
        return features

    def plot(
        self,
        start_time=0,
        width=10,
        use_raw=False,
        scatter=False,
        line=True,
        first_der=False,
        label="",
        use_samples=False,
        ax=None,
        title="",
        lw=3,
    ):
        return_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
            return_fig = True
        if width == -1:
            width = self.time[-1] - start_time
        end_time = start_time + width
        signal_slice = self.get_slice(start_time, end_time)
        if label is not None:
            label = signal_slice.name if label == "" else label
        if use_samples:
            x = np.arange(signal_slice.n_samples) + int(start_time * self.fs)
        else:
            x = signal_slice.time + start_time + signal_slice.start_sec
        y = signal_slice.data if use_raw else signal_slice.cleaned
        plot_kwgs = dict(lw=lw, label=label)
        sig_plot = (
            ax.scatter(x, y, **plot_kwgs, s=self.fig_params["marker_size"]) if scatter else ax.plot(x, y, **plot_kwgs)
        )
        plots = [sig_plot]
        if first_der:
            ax2 = ax.twinx()
            signal_slice_der = signal_slice.get_derivative()
            x = np.arange(signal_slice_der.n_samples) if use_samples else signal_slice_der.time
            y = signal_slice_der.data
            plot_kwgs = dict(c=sns.color_palette()[1], lw=1, label=label + "_der")
            first_der_plot = ax2.scatter(x, y, **plot_kwgs) if scatter else ax2.plot(x, y, **plot_kwgs)
            plots.append(first_der_plot)
            ax2.set_ylabel("First derivative values", fontsize=self.fig_params["label_size"])
            ax2.axhline(y=0, c="black", ls="--", lw=0.5)
        plots = [plot[0] for plot in plots] if line else plots
        labels = [plot.get_label() for plot in plots]
        # ax.legend(plots, labels, loc=0, fontsize=18)
        if label is not None and label != "":
            ax.legend(fontsize=self.fig_params["legend_size"])
        ax.set_ylabel(self.units, fontsize=self.fig_params["label_size"])
        ax.set_xlabel("Samples" if use_samples else "Time [s]", fontsize=self.fig_params["label_size"])
        if title is not None:
            ax.set_title(f"{self.name} {title}", fontsize=self.fig_params["title_size"])
        ax.tick_params(axis="both", which="major", labelsize=self.fig_params["tick_size"])
        # step = int(2 * self.fs) if use_samples else 2
        # if step > x[-1] - x[0]:
        #     step = (x[-1] - x[0]) / 3

        # ax.set_xticks((np.arange(x[0], x[-1] + step / 2, step)).round(2))
        if return_fig:
            return fig

    def explore(self, start_time, width=None):
        if width is None:
            width = self.time[-1] - start_time

        figs = {}
        figs["signal"] = self.plot(start_time=start_time, width=width)
        return figs


class Signal(BaseSignal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.feature_extraction_funcs.update(
            {
                "peaks_troughs": self.extract_peaks_troughs_features,
                "DWT": self.extract_DWT_features,
            }
        )

    def set_windows(self, win_len_s, step_s):
        win_len_samples = win_len_s * self.fs
        step_samples = step_s * self.fs
        intervals = np.array(get_windows(0, self.n_samples, win_len_samples, step_samples))
        if np.diff(intervals[-1]) != np.diff(intervals[0]):
            intervals = intervals[:-1]
        intervals = np.array(intervals).astype(int)
        self.windows = np.array(
            [
                create_new_obj(
                    self, name=f"{self.name}_window_{i}", data=self.data[start:end], start_sec=self.time[start]
                )
                for i, (start, end) in enumerate(intervals)
            ]
        )

    def extract_peaks_troughs_features(self, return_arr=True, plot=False, ax=None):
        peaks_time, peaks_data = self.time[self.peaks], self.cleaned[self.peaks]
        troughs_time, troughs_data = self.time[self.troughs], self.cleaned[self.troughs]
        peaks_new_time = self.time[(self.time >= peaks_time[0]) & (self.time <= peaks_time[-1])]
        peaks_interp = interpolate_to_new_time(peaks_time, peaks_data, new_time=peaks_new_time)
        troughs_interp = interpolate_to_new_time(troughs_time, troughs_data, new_time=peaks_new_time)
        dc = peaks_interp - troughs_interp

        peaks_sig = BaseSignal("peaks", peaks_interp, self.fs, start_sec=peaks_new_time[0])
        troughs_sig = BaseSignal("troughs", troughs_interp, self.fs, start_sec=peaks_new_time[0])
        amplitudes_sig = BaseSignal("amplitudes", dc, self.fs, start_sec=peaks_new_time[0])
        signals = [peaks_sig, troughs_sig, amplitudes_sig]

        features = {sig.name: sig.extract_basic_features(return_arr=False) for sig in signals}
        return_fig = False
        if plot:
            if ax is None:
                return_fig = True
                fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
            for sig in [self] + signals:
                lw = 2 if sig in signals else 3
                sig.plot(start_time=sig.start_sec, width=sig.duration, ax=ax, label=sig.name, lw=lw)
        if return_arr:
            return parse_feats_to_array(features)
        if return_fig:
            return features, fig
        return features

    def extract_DWT_features(self, return_arr=True, wavelet="db8", level=None, plot=False):
        """Discrete wavelet transform"""
        if plot:
            features, fig = extract_dwt_features(self, wavelet, level, plot=True)
        else:
            features = extract_dwt_features(self, wavelet, level, plot=False)
        if return_arr:
            return parse_feats_to_array(features)
        if plot:
            return features, fig
        return features

    def get_CWT(self, wavelet="morl", freq_dim_scale=256, fmin=None, fmax=None, plot=False):
        """Contiunous wavelet transform"""
        freqs, cwt_matrix = get_cwt_img(data=self.cleaned, fs=self.fs, freq_dim_scale=freq_dim_scale, wavelet=wavelet)
        fmin = fmin or min(freqs)
        fmax = fmax or max(freqs)
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs, cwt_matrix = freqs[mask], cwt_matrix[mask]
        if plot:
            fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
            im = ax.contourf(self.time, freqs, cwt_matrix, cmap=plt.cm.seismic)  # PRGn
            ax.set_xlabel("Time [s]", fontsize=self.fig_params["label_size"])
            ax.set_ylabel("Frequency [Hz]", fontsize=self.fig_params["label_size"])
            ax.tick_params(axis="both", which="major", labelsize=self.fig_params["tick_size"])
            cb = fig.colorbar(im)
            cb.ax.set_title("CWT coefficients", fontsize=self.fig_params["label_size"] // 1.5)
            cb.ax.tick_params(labelsize=self.fig_params["tick_size"])

            return freqs, cwt_matrix, fig
        return freqs, cwt_matrix

    def plot_windows(self, **kwargs):
        for window in self.windows:
            window.plot(**kwargs)

    def get_windows_waveforms(
        self, return_arr=True
    ) -> Union[NDArray[Shape["B, N"], Float], Dict[str, NDArray[Shape["N"], Float]]]:
        if return_arr:
            return np.array([window.get_whole_signal_waveform() for window in self.windows])
        return {f"window_{i}": window.get_whole_signal_waveform() for i, window in enumerate(self.windows)}

    def get_windows_features(
        self, return_arr=True
    ) -> Union[NDArray[Shape["B, F"], Float], Dict[str, Dict[str, float]]]:
        if return_arr:
            return np.array([window.extract_features(return_arr=True) for window in self.windows])
        return {f"window_{i}": window.extract_features(return_arr=False) for i, window in enumerate(self.windows)}

    def explore(self, start_time=0, width=None):
        figs = super().explore(start_time, width)
        _, dwt_fig = self.extract_DWT_features(return_arr=False, plot=True)
        _, peaks_troughs_fig = self.extract_peaks_troughs_features(return_arr=False, plot=True)

        figs["DWT"] = dwt_fig
        figs["peaks_troughs"] = peaks_troughs_fig
        return figs


class PeriodicSignal(ABC, Signal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.valid_beats_mask = None
        self.feature_extraction_funcs.update(
            {
                "agg_beat": self.extract_agg_beat_features,
            }
        )

    @property
    def most_dominant_freq(self):
        freqs, amps = self.psd(plot=False)
        most_dominant_freq = freqs[amps.argmax()]
        return most_dominant_freq

    def extract_agg_beat_features(self, return_arr=False, plot=False):
        try:
            return self.agg_beat.extract_features(plot=plot, return_arr=return_arr)
        except AttributeError:
            if return_arr:
                return np.array([])
            else:
                return {}

    @abstractmethod
    def _get_beats_intervals(self, **kwargs):
        return None

    @property
    def BeatClass(self):
        return BeatSignal

    def set_beats(
        self,
        intervals=None,
        align_peaks_loc=None,
        resample=True,
        n_samples=100,
        plot=False,
        use_raw=False,
        normalize=False,
        **kwargs,
    ):
        if intervals is None:
            intervals = self._get_beats_intervals(**kwargs)
        if not resample:
            intervals[:, 1] = intervals[:, 0] + n_samples
        data = self.data if use_raw else self.cleaned
        if align_peaks_loc is not None:
            for i in range(len(intervals)):
                start, end = intervals[i]
                peak_loc = align_peaks_loc[i]
                peak_diff = start + data[start:end].argmax() - peak_loc
                intervals[i][0] = max(0, start + peak_diff)
                intervals[i][1] = min(self.n_samples, end + peak_diff)
        min_beat_n_samples = int(60 / (MAX_HR + 30) * self.fs)
        intervals = np.array(
            [(start, end) for start, end in intervals if (end - start) > min_beat_n_samples]
        )  # TODO Check if works fine
        beats = []
        for i, (start, end) in enumerate(intervals):
            start_sec = start / self.fs
            beat_data = data[start:end]
            if normalize:
                beat_data = z_score(beat_data)
            beat = self.BeatClass(
                name=f"{self.name}_beat_{i}", data=beat_data, fs=self.fs, start_sec=start_sec, beat_num=i
            )
            if resample:
                beat = beat.resample_with_interpolation(n_samples=100, kind="pchip")
            beats.append(beat)
        self.beats = np.array(beats, dtype=self.BeatClass)
        self.validate_beats()
        if plot:
            self.plot_beats()

    def validate_beats(self, max_duration=1.1, IQR_scale=1.5):
        self.valid_beats_mask = get_valid_beats_mask(beats=self.beats, max_duration=max_duration, IQR_scale=IQR_scale)
        for beat, is_valid in zip(self.beats, self.valid_beats_mask):
            beat.is_valid = is_valid

    def set_agg_beat(self, valid_only=False):
        if hasattr(self, "beats"):
            beats_to_aggregate_mask = self.valid_beats_mask if valid_only else len(self.beats) * [True]
            if beats_to_aggregate_mask is None:
                beats_to_aggregate_mask = len(self.beats) * [True]
            beats_to_aggregate = self.beats[beats_to_aggregate_mask]
            beats_data = np.array([beat.data for beat in beats_to_aggregate])
            beats_times = np.array([beat.time for beat in beats_to_aggregate])
            agg_beat_data, agg_beat_time = beats_data.mean(axis=0), beats_times.mean(axis=0)
            agg_fs = len(agg_beat_data) / (agg_beat_time[-1] - agg_beat_time[0])
            self.agg_beat = self.BeatClass(f"{self.name}_agg_beat", agg_beat_data, agg_fs, start_sec=0)
            if hasattr(self, "windows"):
                for window in self.windows:
                    window.beats = np.array(
                        [beat for beat in self.beats if beat.is_in_time_bounds(window.start_sec, window.end_sec)]
                    )
                    window.set_agg_beat()

    def get_beats_features(self, return_arr=True) -> Union[NDArray[Shape["B, F"], Float], Dict[str, Dict[str, float]]]:
        if return_arr:
            return np.array([beat.extract_features(return_arr=True) for beat in self.beats])
        return {f"beat_{beat.beat_num}": beat.extract_features(return_arr=False) for beat in self.beats}

    def get_beats_waveforms(
        self, return_arr=True
    ) -> Union[NDArray[Shape["B, N"], Float], Dict[str, NDArray[Shape["N"], Float]]]:
        if return_arr:
            return np.array([beat.get_whole_signal_waveform() for beat in self.beats])
        return {f"beat_{beat.beat_num}": beat.get_whole_signal_waveform() for beat in self.beats}

    def get_agg_beat_features(self, return_arr=True) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        return self.agg_beat.extract_features(return_arr=return_arr)

    def get_agg_beat_waveform(self) -> NDArray[Shape["N"], Float]:
        return self.agg_beat.data

    def plot_signal_and_agg_beat(self, use_raw=False, axes=None):
        return_fig = False
        if axes is None:
            fig, axes = plt.subplots(
                1, 2, figsize=self.fig_params["fig_size"], sharey=True, gridspec_kw={"width_ratios": [8, 2]}
            )
            return_fig = True
        self.plot(ax=axes[0], use_raw=use_raw, title=None, label=None)
        self.agg_beat.plot(ax=axes[1], title=None, label=None)
        plt.tight_layout()
        if return_fig:
            return fig

    def plot_segmentation(self, mode="windows", use_raw=False, show_lines=True, use_samples=False):
        w, h = self.fig_params["fig_size"]
        fig = plt.figure(figsize=(w, h * 2), layout="constrained")
        signals = self.windows if mode == "windows" else self.beats
        spec = fig.add_gridspec(2, len(signals))
        ax0 = fig.add_subplot(spec[0, :])
        # fig, ax = plt.subplots(1, 1, figsize=self.fig_params["fig_size"])
        self.plot(0, -1, ax=ax0, use_raw=use_raw, title=None, label=None, use_samples=use_samples)
        axes = []
        for j in range(len(signals)):
            axes.append(fig.add_subplot(spec[1, j], sharey=axes[-1] if j != 0 else None))
        axes = np.array(axes)
        data = self.data if use_raw else self.cleaned
        min_v, max_v = data.min(), data.max()
        signals_bounds = []
        color = "green"
        for i, (ax, signal) in enumerate(zip(axes, signals)):
            bounds = [signal.start_sec, signal.end_sec]
            ax0.fill_between(bounds, min_v, max_v, alpha=self.fig_params["fill_alpha"], color=color)
            signals_bounds.extend(bounds)
            color = "green" if color == "red" else "red"
            signal.plot(0, -1, ax=ax, use_raw=use_raw, label=None, title=None, use_samples=use_samples)
            if i != 0:
                ax.set_ylabel("")
                plt.setp(ax.get_yticklabels(), visible=False)
        if show_lines:
            ax0.vlines(signals_bounds, min_v, max_v, lw=1.5, alpha=0.6, ec="black", ls="--")
        ax0.set_xlim([self.time[0], self.time[-1]])
        return fig

    def plot_beats_segmentation(
        self, valid=True, invalid=True, use_raw=False, color_validity=True, show_beat_lines=True, axes=None
    ):
        return_fig = False
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=self.fig_params["fig_size"], gridspec_kw={"width_ratios": [8, 2]})
            return_fig = True
        if use_raw:
            self.plot(ax=axes[0], title="beats segmentation")
            data = self.data
        else:
            axes[0].plot(self.time, self.cleaned, lw=3, label=self.name)
            data = self.cleaned
        beats_bounds = []
        color = "green"
        for beat in self.beats:
            bounds = [beat.start_sec, beat.end_sec]
            if color_validity is not None and color_validity:
                color = "green" if beat.is_valid else "red"
                label = "valid" if beat.is_valid else "invalid"
            else:
                color = "green"  # if color == "red" else "red"
                label = None
            if color_validity is not None:
                axes[0].fill_between(
                    bounds, data.min(), data.max(), alpha=self.fig_params["fill_alpha"], color=color, label=label
                )
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[0].legend(by_label.values(), by_label.keys(), fontsize=self.fig_params["legend_size"])

            beats_bounds.extend(bounds)
        if show_beat_lines:
            axes[0].vlines(beats_bounds, data.min(), data.max(), lw=1.5, alpha=0.6, ec="black", ls="--")
        self.plot_beats(valid=valid, invalid=invalid, ax=axes[1])
        axes[0].set_xlim([self.time.min(), self.time.max()])
        axes[0].grid(False)
        for ax in axes:
            ax.set_xlabel("Time [s]", fontsize=self.fig_params["label_size"])
            ax.set_ylabel(self.units, fontsize=self.fig_params["label_size"])
            ax.tick_params(axis="both", which="major", labelsize=self.fig_params["tick_size"])
        plt.tight_layout()
        if return_fig:
            return fig

    def plot_beats(self, same_ax=True, ax=None, valid=True, invalid=True):
        beats_to_plot = [beat for beat in self.beats if (beat.is_valid and valid) or (not beat.is_valid and invalid)]

        n_beats = len(beats_to_plot)
        n_valid = np.sum([beat.is_valid for beat in beats_to_plot])
        n_invalid = n_beats - n_valid
        valid_palette = sns.color_palette("Greens", n_colors=n_beats)
        invalid_palette = sns.color_palette("Reds", n_colors=n_beats)

        if same_ax:
            if ax is None:
                fig, ax = plt.subplots(figsize=self.agg_beat.fig_params["fig_size"])
            for i, beat in enumerate(beats_to_plot):
                palette = valid_palette if beat.is_valid else invalid_palette
                ax.plot(beat.time, beat.data, color=palette[i])
            agg_beat = self.agg_beat
            ax.plot(agg_beat.time, agg_beat.data, lw=8, alpha=0.5, c="black", label="Aggregated beat")
            if valid is False or invalid is False:
                ax.set_title(
                    f"{n_beats} beats ({n_valid} valid, {n_invalid} invalid)",
                    fontsize=int(agg_beat.fig_params["title_size"] / 1.5),
                )
            ax.set_ylabel(agg_beat.units, fontsize=agg_beat.fig_params["label_size"])
            ax.set_xlabel("Time [s]", fontsize=agg_beat.fig_params["label_size"])
            ax.tick_params(axis="both", which="major", labelsize=self.fig_params["tick_size"])
        else:
            ncols = 5
            nrows = int(np.ceil(n_beats / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            for beat, ax in zip(beats_to_plot, axes.flatten()):
                palette = valid_palette if beat.is_valid else invalid_palette
                ax.plot(beat.time, beat.data, lw=3, c=palette[int(n_beats / 1.5)])
                ax.set_title(beat.name, fontsize=beat.fig_params["title_size"])

    def explore(self, start_time=0, width=None):
        figs = super().explore(start_time, width)
        beats_segmentation_fig = self.plot_beats_segmentation(show_beat_lines=False)
        figs["beats_segmentation"] = beats_segmentation_fig
        return figs


class BeatSignal(ABC, BaseSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0, is_valid=True):
        super().__init__(name, data, fs, start_sec)
        self.is_valid = is_valid
        self.beat_num = beat_num
        self.fig_params = BEAT_FIG_PARAMS
        self.feature_extraction_funcs.update(
            {
                "crit_points": self.extract_crit_points_features,
                "area": self.extract_area_features,
                "slope": self.extract_slope_features,
                "energy": self.extract_energy_features,
                "intervals": self.extract_intervals_features,
            }
        )

    def extract_basic_features(self, return_arr=True, **kwargs) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = super().extract_basic_features(return_arr=False)
        features.update(
            {
                "duration": self.duration,
                "max": self.max,
                "min": self.min,
            }
        )
        if return_arr:
            return np.array(list(features.values()))
        return features

    @lazy_property
    def crit_points(self) -> Dict[str, float]:
        return {}

    @lazy_property
    def energy_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        """Must return list of dicts with `name`, `start` and `end` keys

        Example: [{"name": <feat_name>, "start": <start_location>, "end": <end_location>}, ...]
        """
        return []

    @lazy_property
    def area_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        """Must return list of dicts with `name`, `start` and `end` keys

        Example: [{"name": <feat_name>, "start": <start_location>, "end": <end_location>}, ...]
        """
        return []

    @lazy_property
    def slope_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        """Must return list of dicts with `name`, `start` and `end` keys

        Example: [{"name": <feat_name>, "start": <start_location>, "end": <end_location>}, ...]
        """
        return []

    @lazy_property
    def intervals_features_crit_points(self) -> List[Dict[str, Union[int, str]]]:
        """Must return list of dicts with `name`, `start` and `end` keys

        Example: [{"name": <feat_name>, "start": <start_location>, "end": <end_location>}, ...]
        """
        return []

    def extract_crit_points_features(
        self, return_arr=True, plot=False, ax=None
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = {}
        for name, loc in self.crit_points.items():
            features[f"{name}_loc"] = loc
            features[f"{name}_time"] = self.time[loc]
            features[f"{name}_val"] = self.cleaned[loc]
        if plot:
            fig = self.plot_crit_points(plot_sig=True, ax=ax)
        if return_arr:
            return parse_feats_to_array(features)
        if plot:
            return features, fig
        return features

    def extract_energy_features(self, return_arr=True, **kwargs) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = {
            point["name"]: calculate_energy(self.cleaned, point["start"], point["end"])
            for point in self.energy_features_crit_points
        }
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_area_features(
        self, return_arr=True, method="trapz", plot=False, **kwargs
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = {
            point["name"]: calculate_area(self.cleaned, self.fs, point["start"], point["end"], method=method)
            for point in self.area_features_crit_points
        }
        if plot:
            fig = self.plot_area_features()
        if return_arr:
            return parse_feats_to_array(features)
        if plot:
            return features, fig
        return features

    def extract_slope_features(
        self, return_arr=True, plot=False, ax=None
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = {
            point["name"]: calculate_slope(self.time, self.cleaned, point["start"], point["end"])
            for point in self.slope_features_crit_points
        }
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_intervals_features(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        time_features = {
            f"{point['name']}_time": (point["end"] - point["start"]) / self.fs
            for point in self.intervals_features_crit_points
        }

        val_diff_features = {
            f"{point['name']}_val_diff": self.cleaned[point["end"]] - self.cleaned[point["start"]]
            for point in self.intervals_features_crit_points
        }
        features = {**time_features, **val_diff_features}
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def plot_crit_points(self, plot_sig=False, ax=None, crit_points="all", **kwargs):
        t = self.time
        t_range = t[-1] - t[0]
        offset = 0
        return_fig = False
        if ax is None:
            return_fig = True
            fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
        if plot_sig:
            self.plot(ax=ax, title="crit points")
        if crit_points == "all":
            crit_points = list(self.crit_points.keys())
        for name, loc in self.crit_points.items():
            if name not in crit_points:
                continue
            t_loc = t[loc]
            val = self.cleaned[loc] + offset
            ax.scatter(t_loc, val, label=name, s=self.fig_params["marker_size"])
            ax.annotate(
                name.upper(),
                (t_loc + 0.012 * t_range, val + 0.012 * self.range),
                fontweight="bold",
                fontsize=self.fig_params["annot_size"],
            )
        if return_fig:
            return fig

    def plot_area_features(self, ax=None, crit_points="all"):
        return_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
            return_fig = True
        palette = sns.color_palette("pastel", len(self.area_features_crit_points))
        if crit_points == "all":
            crit_points = list(self.crit_points.keys())
        self.plot_crit_points(ax=ax, crit_points=crit_points)
        self.plot(ax=ax, title="area features", label=None)
        areas = []
        for i, point in enumerate(self.area_features_crit_points):
            offset = min([self.cleaned[point["start"]], self.cleaned[point["end"]]])
            mask = (np.arange(self.n_samples) >= point["start"]) & (np.arange(self.n_samples) <= point["end"])
            a = ax.fill_between(
                self.time[mask],
                self.cleaned[mask],
                offset,
                alpha=0.4,
                color=palette[i],
                edgecolor="black",
                lw=1,
                label=point["name"],
            )
            areas.append(a)
        leg = ax.legend(handles=areas, fontsize=self.fig_params["legend_size"], loc=1)
        ax.add_artist(leg)
        if return_fig:
            return fig

    def explore(self, start_time=0, width=None):
        figs = super().explore(start_time, width)
        _, area_features_fig = self.extract_area_features(return_arr=False, plot=True)
        _, crit_points_fig = self.extract_crit_points_features(return_arr=False, plot=True)
        figs["area_features"] = area_features_fig
        figs["crit_points"] = crit_points_fig
        return figs


class MultiChannelSignal:
    def __init__(self, signals: Dict[str, Signal]):
        self.signals = signals
        self.n_signals = len(signals)

        self.representations_funcs = {
            "whole_signal_waveforms": self.get_whole_signal_waveforms,
            "whole_signal_features": self.get_whole_signal_features,
            "windows_waveforms": self.get_windows_waveforms,
            "windows_features": self.get_windows_features,
        }

        self.feature_names_funcs = {
            "whole_signal_features": self.get_whole_signal_feature_names,
            "windows_features": self.get_windows_feature_names,
        }

        signals_extraction_funcs = [{name: sig.extract_features for name, sig in self.signals.items()}]
        self.feature_extraction_funcs = ChainMap(*signals_extraction_funcs)

    def __getitem__(self, key):
        return self.signals[key]

    def extract_features(self, return_arr=False, plot=False) -> Union[NDArray[Shape["FC"], Float], Dict[str, float]]:
        features = {name: func(return_arr=False, plot=plot) for name, func in self.feature_extraction_funcs.items()}
        features = parse_nested_feats(features)
        self.feature_names = list(features.keys())
        if return_arr:
            return np.array(list(features.values()))
        return features

    def set_windows(self, win_len_s, step_s, **kwargs):
        self.windows = {name: sig.set_windows(win_len_s, step_s) for name, sig in self.signals.items()}

    def get_whole_signal_waveforms(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["N, C"], Float], Dict[str, NDArray[Shape["N"], Float]]]:
        """Return whole signal waveforms representation, i.e. timeseries signals data.

        N - number of samples in a signal
        C - number of channels

        If return_arr is True, then `NDArray[Shape["N, C"], Float]` is returned
        If return_arr if False, then signal for each channel is returned in dict form (`Dict[str, NDArray[Shape["N"], Float]]`)
        """
        if return_arr:
            return np.array([sig.get_whole_signal_waveform() for name, sig in self.signals.items()]).T
        return {name: sig.get_whole_signal_waveform() for name, sig in self.signals.items()}

    def get_whole_signal_features(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]:
        """Return whole signal features representation

        F - number of features from single signal
        C - number of channels

        If return_arr is True, then `NDArray[Shape["F, C"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]`)
        """
        if return_arr:
            # return np.array(
            #     [func(return_arr=True, plot=False) for name, func in self.feature_extraction_funcs.items()]
            # ).T
            return self.extract_features(return_arr=True)
        return {name: func(return_arr=False) for name, func in self.feature_extraction_funcs.items()}

    def get_windows_waveforms(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["W, N, C"], Float], Dict[str, Dict[str, NDArray[Shape["N"], Float]]]]:
        """Return windows waveforms representation, i.e. timeseries data for each window.

        B - number of windows
        N - number of samples in a window
        C - number of channels

        If return_arr is True, then `NDArray[Shape["B, N, C"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["N"], Float]]]]`)
        """
        if return_arr:
            return np.array([sig.get_windows_waveforms(return_arr=True) for _, sig in self.signals.items()]).transpose(
                1, 2, 0
            )
        return {name: sig.get_windows_waveforms(return_arr=False) for name, sig in self.signals.items()}

    def get_windows_features(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["B, F, C"], Float], Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]:
        """Return windows features representation, i.e. features extracted for each window.

        B - number of windows
        F - number of features in a window
        C - number of channels

        If return_arr is True, then `NDArray[Shape["B, F, C"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]`)
        """
        if return_arr:
            return np.concatenate(
                [sig.get_windows_features(return_arr=True) for _, sig in self.signals.items()], axis=1
            )
        return {name: sig.get_windows_features(return_arr=False) for name, sig in self.signals.items()}

    def get_whole_signal_feature_names(self):
        self.extract_features()
        return np.array(self.feature_names)

    def get_windows_feature_names(self):
        n_windows = len(list(self.signals.values())[0].windows)
        all_feature_names = []
        for name, sig in self.signals.items():
            sig.windows[0].extract_features()
        for i in range(n_windows):
            sig_feature_names = []
            for name, sig in self.signals.items():
                window_sig = sig.windows[i]
                feature_names = sig.windows[0].feature_names
                feature_names = [f"{window_sig.name}__{feature_name}" for feature_name in feature_names]
                sig_feature_names.extend(feature_names)
            all_feature_names.append(sig_feature_names)
        return np.array(all_feature_names)

    def plot(
        self,
        start_time=0,
        width=10,
        scatter=False,
        line=True,
        first_der=False,
        label=None,
        use_samples=False,
        axes=None,
        title="",
    ):
        kwargs = dict_of(start_time, width, scatter, line, first_der, label, use_samples, title)
        return_fig = False
        w, h = SIGNAL_FIG_PARAMS["fig_size"]
        if axes is None:
            return_fig = True
            fig, axes = plt.subplots(
                self.n_signals, 1, figsize=(w, 1.6 * self.n_signals + 1 / self.n_signals), sharex=True
            )
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot(ax=ax, **kwargs)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(sig_name, fontsize=sig.fig_params["title_size"])
            ax.tick_params(axis="both", which="major", labelsize=sig.fig_params["tick_size"])
            # ax.grid(False)
            # ax.get_legend().remove()
        axes[-1].set_xlabel("Time [s]", fontsize=sig.fig_params["label_size"])

        plt.tight_layout()
        if return_fig:
            return fig


class MultiChannelPeriodicSignal(MultiChannelSignal):
    def __init__(self, signals: Dict[str, PeriodicSignal]):
        super().__init__(signals)

        self.representations_funcs.update(
            {
                "beats_waveforms": self.get_beats_waveforms,
                "beats_features": self.get_beats_features,
                "agg_beat_waveforms": self.get_agg_beat_waveforms,
                "agg_beat_features": self.get_agg_beat_features,
            }
        )

        self.feature_names_funcs.update(
            {
                "beats_features": self.get_beats_feature_names,
                "agg_beat_features": self.get_agg_beat_feature_names,
            }
        )

    def set_beats(self, source_channel=None, align_peaks_loc=False, **kwargs):
        """Return beats from all channels.

        If `source_channel` is specified, it will be used as source of beat intervals
        for other channels. If `ZeroDivisionError` occurs (from `neurokit2`) for specified source_channel,
        other channels are tested.
        If `source_channel` is `None`, beats are extracter for every channel separately. In that case, there
        is no guarantee, that number of beats and their alignment is preserved across all channels.

        Args:
            source_channel (int, optional): Channel used as source of beat intervals. Defaults to `2`.
        """
        if source_channel is not None:
            source_channels = list(self.signals.keys())
            source_channels.remove(source_channel)
            source_channels.reverse()
            found_good_channel = False
            new_source_channel = source_channel
            while not found_good_channel and len(source_channels) >= 0:
                try:
                    source_signal = self.signals[new_source_channel]
                    source_signal.set_beats(**kwargs)
                    min_n_beats, max_n_beats = (
                        MIN_HR / 60 * source_signal.duration,
                        MAX_HR / 60 * source_signal.duration,
                    )
                    if min_n_beats <= len(source_signal.beats) <= max_n_beats:
                        found_good_channel = True
                        if new_source_channel != source_channel:
                            log.info(
                                f"Original source channel was corrupted. Found new source channel: {new_source_channel}"
                            )
                    else:
                        log.error("Number of beats indicates that HR bounds werent met. Trying another channel")
                        raise Exception("Number of beats indicates that HR bounds werent met. Trying another channel")
                except Exception as e:
                    new_source_channel = source_channels.pop()
            if not found_good_channel:
                new_source_channel = source_channel
                self.signals[new_source_channel].set_beats(**kwargs)  # use Heart Rate (most dominant frequency)
            intervals = self.signals[new_source_channel]._get_beats_intervals()
        else:
            intervals = None
        for name, signal in self.signals.items():
            if name == new_source_channel:
                continue
            if align_peaks_loc:
                _align_peaks_loc = source_signal.peaks
                if len(_align_peaks_loc) != len(intervals):
                    _align_peaks_loc = np.array(
                        [source_signal.cleaned[start:end].argmax() + start for start, end in intervals]
                    )
            else:
                _align_peaks_loc = None
            signal.set_beats(intervals, align_peaks_loc=_align_peaks_loc, **kwargs)

    def set_agg_beat(self, **kwargs):
        for _, signal in self.signals.items():
            signal.set_agg_beat(**kwargs)

    def get_beats_features(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["B, F, C"], Float], Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]:
        """Return beats features representation, i.e. features extracted for each beat.

        B - number of beats
        F - number of features in a beat
        C - number of channels

        If return_arr is True, then `NDArray[Shape["B, F, C"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]`)
        """

        if return_arr:
            return np.concatenate([sig.get_agg_beat_features(return_arr=True) for _, sig in self.signals.items()])
        return {name: sig.get_beats_features(return_arr=False) for name, sig in self.signals.items()}

    def get_beats_waveforms(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["W, N, C"], Float], Dict[str, Dict[str, NDArray[Shape["N"], Float]]]]:
        """Return beats waveforms representation, i.e. timeseries data for each beat.

        B - number of beats
        N - number of samples in a beat
        C - number of channels

        If return_arr is True, then `NDArray[Shape["B, N, C"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["N"], Float]]]]`)
        """
        if return_arr:
            return np.array([sig.get_beats_waveforms(return_arr=True) for _, sig in self.signals.items()]).transpose(
                1, 2, 0
            )
        return {name: sig.get_beats_waveforms(return_arr=False) for name, sig in self.signals.items()}

    def get_agg_beat_waveforms(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["N, C"], Float], Dict[str, NDArray[Shape["N"], Float]]]:
        """Return aggregated beat waveforms representation, i.e. timeseries data for each aggregated beat.

        N - number of samples in an aggregated beat.
        C - number of channels

        If return_arr is True, then `NDArray[Shape["N, C"], Float]` is returned
        If return_arr if False, then signal for each channel is returned in dict form (`Dict[str, NDArray[Shape["N"], Float]]`)
        """
        if return_arr:
            return np.array([sig.get_agg_beat_waveform() for _, sig in self.signals.items()]).T
        return {name: sig.get_agg_beat_waveform() for name, sig in self.signals.items()}

    def get_agg_beat_features(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]:
        """Return aggregated beat features representation, i.e. features extracted for each aggregated beat.

        F - number of features from single aggregated beat.
        C - number of channels

        If return_arr is True, then `NDArray[Shape["F, C"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]`)
        """
        if return_arr:
            return np.concatenate([sig.get_agg_beat_features(return_arr=True) for _, sig in self.signals.items()])
        return {name: sig.get_agg_beat_features(return_arr=False) for name, sig in self.signals.items()}

    def get_beats_feature_names(self):
        n_beats = len(list(self.signals.values())[0].beats)
        all_feature_names = []
        for name, sig in self.signals.items():
            sig.beats[0].extract_features()
        for i in range(n_beats):
            sig_feature_names = []
            for name, sig in self.signals.items():
                beat_sig = sig.beats[i]
                feature_names = sig.beats[0].feature_names
                feature_names = [f"{beat_sig.name}__{feature_name}" for feature_name in feature_names]
                sig_feature_names.extend(feature_names)
            all_feature_names.append(sig_feature_names)
        return np.array(all_feature_names)

    def get_agg_beat_feature_names(self):
        all_feature_names = []
        for name, sig in self.signals.items():
            sig.extract_agg_beat_features()
            feature_names = sig.agg_beat.feature_names
            feature_names = [f"{sig.agg_beat.name}__{feature_name}" for feature_name in feature_names]
            all_feature_names.extend(feature_names)
        return np.array(all_feature_names)

    def plot_beats_segmentation(self, valid=True, invalid=True, use_raw=False, color_validity=True, **kwargs):
        w, h = SIGNAL_FIG_PARAMS["fig_size"]
        fig, all_axes = plt.subplots(
            self.n_signals,
            2,
            figsize=(w, 1.6 * self.n_signals + 1 / self.n_signals),
            sharex="col",
            gridspec_kw={"width_ratios": [8, 2]},
        )
        for axes, (sig_name, sig) in zip(all_axes, self.signals.items()):
            sig.plot_beats_segmentation(
                valid=valid, invalid=invalid, use_raw=use_raw, color_validity=color_validity, axes=axes
            )
            for ax in axes:
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="both", which="major", labelsize=sig.fig_params["tick_size"])
            axes[0].set_title(sig_name, fontsize=sig.fig_params["title_size"])
            axes[0].get_legend().remove()
            # ax.grid(False)
            # ax.get_legend().remove()
        for ax in all_axes[-1]:
            ax.set_xlabel("Time [s]", fontsize=sig.fig_params["label_size"])

        plt.tight_layout()

        return fig
