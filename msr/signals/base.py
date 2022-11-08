from __future__ import annotations

from abc import ABC, abstractmethod
from collections import ChainMap, OrderedDict
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
from scipy.stats import kurtosis, skew
from statsmodels.tsa.seasonal import STL

from msr.signals.utils import (
    BEAT_FIG_PARAMS,
    SIGNAL_FIG_PARAMS,
    calculate_area,
    calculate_energy,
    calculate_slope,
    get_valid_beats_mask,
    get_windows,
    parse_feats_to_array,
    parse_nested_feats,
    z_score,
)
from msr.utils import create_new_obj, lazy_property

MIN_HR, MAX_HR = 30, 200


class BaseSignal:
    def __init__(self, name: str, data: NDArray[Shape["N"], Float], fs: float, start_sec: float = 0):
        self.name = name
        self.data = data
        self.fs = fs
        self.start_sec = start_sec
        self.n_samples = len(data)
        self.duration = self.n_samples / fs
        self.time = np.arange(0, self.n_samples) / fs
        self.end_sec = self.start_sec + self.time[-1]
        self.fig_params = SIGNAL_FIG_PARAMS
        self.feature_extraction_funcs = {
            "basic_features": self.extract_basic_features,
        }

    @property
    def cleaned(self):
        return np.copy(self.data)

    def find_peaks(self):
        return find_peaks(self.data)[0]

    def find_troughs(self):
        return find_peaks(-self.data)[0]

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
        new_name = self.name + f"_resampleWithInterp({n_samples})"
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
        amps = rfft(self.data)
        freqs = rfftfreq(self.n_samples, 1 / self.fs)
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(freqs[1:], np.abs(amps[1:]))
        return freqs, amps

    def psd(
        self, window_size, min_hz=0, max_hz=10, plot=False, ax=None
    ) -> Tuple[NDArray[Shape["N"], Float], NDArray[Shape["N"], Float]]:
        freqs, amps = welch(self.data, self.fs, nperseg=self.fs * window_size)
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(16, 8))
            mask = (freqs >= min_hz) & (freqs <= max_hz)
            ax.plot(freqs[mask], amps[mask], lw=2)
            ax.set_xlabel("Frequency [Hz]", fontsize=18)
            ax.set_ylabel("Power Spectral Density", fontsize=18)

        return freqs, amps

    def spectrogram(
        self, NFFT=256, noverlap=128, ax=None
    ) -> Tuple[NDArray[Shape["N"], Float], NDArray[Shape["N"], Float], NDArray[Shape["N"], Float]]:
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 8))
        spectrum, freqs, t, img = ax.specgram(self.data, Fs=self.fs, NFFT=NFFT, noverlap=noverlap, cmap="viridis")
        cbar = plt.gcf().colorbar(img, ax=ax)
        cbar.ax.set_title("Intensity", fontsize=14)
        ax.set_ylabel("Frequency [Hz]", fontsize=18)
        ax.set_xlabel("Time [s]", fontsize=18)
        return spectrum, freqs, t

    def get_whole_signal_waveform(self) -> NDArray[Shape["N"], Float]:
        return self.data

    def extract_basic_features(self, return_arr=True, **kwargs) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = OrderedDict(
            {
                "mean": self.data.mean(),
                "std": self.data.std(),
                "median": np.median(self.data),
                "skewness": skew(self.data),
                "kurtosis": kurtosis(self.data),
            }
        )
        if return_arr:
            return np.array(list(features.values()))
        return features

    def extract_features(self, return_arr=True, plot=False) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = OrderedDict(
            {
                f"{self.name}_{name}": func(return_arr=False, plot=plot)
                for name, func in self.feature_extraction_funcs.items()
            }
        )
        features = parse_nested_feats(features)
        self.feature_names = list(features.keys())
        if return_arr:
            return np.array(list(features.values()))
        return features

    def plot(
        self,
        start_time=0,
        width=10,
        scatter=False,
        line=True,
        first_der=False,
        label=None,
        use_samples=False,
        ax=None,
        title="",
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
        if width == -1:
            width = self.time[-1] - start_time
        end_time = start_time + width

        if end_time - start_time >= self.duration:
            signal_slice = self
        else:
            signal_slice = self.get_slice(start_time, end_time)
        label = label if label is not None else signal_slice.name
        x = np.arange(signal_slice.n_samples) if use_samples else signal_slice.time
        y = signal_slice.data
        plot_kwgs = dict(lw=3, label=label)
        sig_plot = ax.scatter(x, y, **plot_kwgs) if scatter else ax.plot(x, y, **plot_kwgs)
        plots = [sig_plot]
        if first_der:
            ax2 = ax.twinx()
            signal_slice_der = signal_slice.get_derivative()
            x = np.arange(signal_slice_der.n_samples) if use_samples else signal_slice_der.time
            y = signal_slice_der.data
            plot_kwgs = dict(c=sns.color_palette()[1], lw=1, label=label + "_der")
            first_der_plot = ax2.scatter(x, y, **plot_kwgs) if scatter else ax2.plot(x, y, **plot_kwgs)
            plots.append(first_der_plot)
            ax2.set_ylabel("First derivative values", fontsize=18)
            ax2.axhline(y=0, c="black", ls="--", lw=0.5)
        plots = [plot[0] for plot in plots] if line else plots
        labels = [plot.get_label() for plot in plots]
        ax.legend(plots, labels, loc=0, fontsize=18)
        ax.set_ylabel("Values", fontsize=18)
        ax.set_xlabel("Samples" if use_samples else "Time [s]", fontsize=18)
        ax.set_title(f"{self.name} {title}", fontsize=22)

    def explore(self, start_time, width=None, window_size=4, min_hz=0, max_hz=20):
        if width is None:
            width = self.time[-1] - start_time

        signal_slice = self.get_slice(start_time, start_time + width)
        fig = plt.figure(figsize=(40, 20), constrained_layout=True)
        spec = fig.add_gridspec(8, 8, wspace=0.8, hspace=0.8)

        ax00 = fig.add_subplot(spec[:4, :6])
        ax01 = fig.add_subplot(spec[:4, 6:])
        ax11 = fig.add_subplot(spec[4:, :4])
        ax12 = fig.add_subplot(spec[4:, 4:])

        signal_slice.plot(start_time, width, ax=ax00)
        sns.kdeplot(signal_slice.data, ax=ax01, lw=3)
        signal_slice.psd(window_size=window_size, min_hz=min_hz, max_hz=max_hz, plot=True, ax=ax11)
        signal_slice.spectrogram(ax=ax12)

        axes_params = [
            (ax00, "Signal"),
            (ax01, "Kernel Density Estimation"),
            (ax11, "Power Spectral Density"),
            (ax12, "Spectrogram"),
        ]

        for ax, title in axes_params:
            ax.set_title(title, fontsize=26, fontweight="bold")
            ax.tick_params(axis="both", which="major", labelsize=18)
            ax.margins(0.01)


class Signal(BaseSignal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)

    def set_windows(self, win_len_s, step_s):
        win_len_samples = win_len_s * self.fs
        step_samples = step_s * self.fs
        intervals = np.array(get_windows(0, self.n_samples, win_len_samples, step_samples))
        if np.diff(intervals[-1]) != np.diff(intervals[0]):
            intervals = intervals[:-1]
        intervals = np.array(intervals).astype(int)
        self.windows = np.array(
            [create_new_obj(self, data=self.data[start:end], start_sec=self.time[start]) for start, end in intervals]
        )

    def plot_windows(self, **kwargs):
        for window in self.windows:
            window.plot(**kwargs)

    def get_windows_waveforms(
        self, return_arr=True
    ) -> Union[NDArray[Shape["B, N"], Float], Dict[str, NDArray[Shape["N"], Float]]]:
        if return_arr:
            return np.array([window.get_whole_signal_waveform() for window in self.windows])
        return OrderedDict({f"window_{i}": window.get_whole_signal_waveform() for i, window in enumerate(self.windows)})

    def get_windows_features(
        self, return_arr=True
    ) -> Union[NDArray[Shape["B, F"], Float], Dict[str, Dict[str, float]]]:
        if return_arr:
            return np.array([window.extract_features(return_arr=True) for window in self.windows])
        return OrderedDict(
            {f"window_{i}": window.extract_features(return_arr=False) for i, window in enumerate(self.windows)}
        )


class PeriodicSignal(ABC, Signal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.valid_beats_mask = None
        self.feature_extraction_funcs.update({"agg_beat_features": self.extract_agg_beat_features})

    @property
    def hr(self):
        freqs, amps = self.psd(window_size=10, plot=False)
        most_dominant_freq = freqs[amps.argmax()]
        return most_dominant_freq

    @abstractmethod
    def extract_agg_beat_features(self, return_arr=False, plot=False):
        pass

    @abstractmethod
    def _get_beats_intervals(self, **kwargs):
        return None

    @property
    def BeatClass(self):
        return BeatSignal

    def set_beats(self, intervals=None, resample=True, n_samples=100, plot=False, use_raw=False, **kwargs):
        if intervals is None:
            intervals = self._get_beats_intervals(**kwargs)
        if not resample:
            intervals[:, 1] = intervals[:, 0] + n_samples
        data = self.data if use_raw else self.cleaned
        beats = []
        for i, (start, end) in enumerate(intervals):
            start_sec = start / self.fs
            beat = self.BeatClass(name=f"beat_{i}", data=data[start:end], fs=self.fs, start_sec=start_sec, beat_num=i)
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
        return OrderedDict({f"beat_{beat.beat_num}": beat.extract_features(return_arr=False) for beat in self.beats})

    def get_beats_waveforms(
        self, return_arr=True
    ) -> Union[NDArray[Shape["B, N"], Float], Dict[str, NDArray[Shape["N"], Float]]]:
        if return_arr:
            return np.array([beat.get_whole_signal_waveform() for beat in self.beats])
        return OrderedDict({f"beat_{beat.beat_num}": beat.get_whole_signal_waveform() for beat in self.beats})

    def get_agg_beat_features(self, return_arr=True) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        return self.agg_beat.extract_features(return_arr=return_arr)

    def get_agg_beat_waveform(self) -> NDArray[Shape["N"], Float]:
        return self.agg_beat.data

    def plot_beats_segmentation(self, use_raw=False, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(24, 4), gridspec_kw={"width_ratios": [8, 2]})
        if use_raw:
            self.plot(ax=axes[0], title="beats segmentation")
        else:
            axes[0].plot(self.time, self.cleaned, lw=3, label=self.name)
        beats_bounds = []
        for beat in self.beats:
            bounds = [beat.start_sec, beat.end_sec]
            color = "green" if beat.is_valid else "red"
            axes[0].fill_between(bounds, self.data.min(), self.data.max(), alpha=0.15, color=color)
            beats_bounds.extend(bounds)
        axes[0].vlines(beats_bounds, self.data.min(), self.data.max(), lw=1.5, ec="black", ls="--")
        self.plot_beats(ax=axes[1])
        axes[0].set_xlim([self.time.min(), self.time.max()])
        axes[0].set_ylim([self.data.min(), self.data.max()])
        for ax in axes:
            ax.grid(False)
            ax.set_xlabel("")
            ax.set_ylabel("")
        plt.tight_layout()

    def plot_beats(self, same_ax=True, ax=None, valid=True, invalid=True):
        beats_to_plot = [beat for beat in self.beats if (beat.is_valid and valid) or (not beat.is_valid and invalid)]

        n_beats = len(beats_to_plot)
        n_valid = np.sum([beat.is_valid for beat in beats_to_plot])
        n_invalid = n_beats - n_valid
        valid_palette = sns.color_palette("Greens", n_colors=n_beats)
        invalid_palette = sns.color_palette("Reds", n_colors=n_beats)

        if same_ax:
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 5))
            for i, beat in enumerate(beats_to_plot):
                palette = valid_palette if beat.is_valid else invalid_palette
                ax.plot(beat.time, beat.data, color=palette[i])
            ax.plot(self.agg_beat.time, self.agg_beat.data, lw=8, alpha=0.5, c="black", label="Aggregated beat")
            ax.set_title(f"{n_beats} beats ({n_valid} valid, {n_invalid} invalid)", fontsize=22)
            ax.set_ylabel("Values", fontsize=18)
            ax.set_xlabel("Time [s]", fontsize=18)
        else:
            ncols = 5
            nrows = int(np.ceil(n_beats / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            for beat, ax in zip(beats_to_plot, axes.flatten()):
                palette = valid_palette if beat.is_valid else invalid_palette
                ax.plot(beat.time, beat.data, lw=3, c=palette[int(n_beats / 1.5)])
                ax.set_title(beat.name, fontsize=18)
            plt.tight_layout()

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20):
        super().explore(start_time, width, window_size, min_hz, max_hz)
        self.plot_beats_segmentation()
        fig, axes = plt.subplots(1, 2, figsize=(24, 5))
        self.plot_beats(ax=axes[0])
        self.agg_beat.plot(with_crit_points=True, ax=axes[1])


class BeatSignal(ABC, BaseSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0, is_valid=True):
        super().__init__(name, data, fs, start_sec)
        self.is_valid = is_valid
        self.beat_num = beat_num
        self.fig_params = BEAT_FIG_PARAMS
        self.feature_extraction_funcs.update(
            {
                "crit_points": self.extract_crit_points_features,
            }
        )

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

    def extract_crit_points_features(
        self, return_arr=True, plot=False, ax=None
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = OrderedDict()
        for name, loc in self.crit_points.items():
            features[f"{name}_loc"] = loc
            features[f"{name}_time"] = self.time[loc]
            features[f"{name}_val"] = self.data[loc]
        if plot:
            self.plot_crit_points(plot_sig=True, ax=ax)
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_energy_features(self, return_arr=True, **kwargs) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = OrderedDict(
            {
                point["name"]: calculate_energy(self.data, point["start"], point["end"])
                for point in self.energy_features_crit_points
            }
        )
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_area_features(
        self, return_arr=True, method="trapz", plot=False, **kwargs
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = OrderedDict(
            {
                point["name"]: calculate_area(self.data, self.fs, point["start"], point["end"], method=method)
                for point in self.area_features_crit_points
            }
        )
        if plot:
            self.plot_area_features()
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def extract_slope_features(
        self, return_arr=True, plot=False, ax=None
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, float]]:
        features = OrderedDict(
            {
                point["name"]: calculate_slope(self.time, self.data, point["start"], point["end"])
                for point in self.slope_features_crit_points
            }
        )
        if return_arr:
            return parse_feats_to_array(features)
        return features

    def plot_crit_points(self, plot_sig=False, ax=None, **kwargs):
        t = self.time
        t_range = t[-1] - t[0]
        y_range = self.data.max() - self.data.min()
        offset = 0
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
        if plot_sig:
            self.plot(ax=ax, title="crit points")
        for name, loc in self.crit_points.items():
            t_loc = t[loc]
            val = self.data[loc] + offset
            ax.scatter(t_loc, val, label=name, s=self.fig_params["marker_size"])
            ax.annotate(
                name.upper(),
                (t_loc + 0.012 * t_range, val + 0.012 * y_range),
                fontweight="bold",
                fontsize=self.fig_params["annot_size"],
            )

    def plot_area_features(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=self.fig_params["fig_size"])
        palette = sns.color_palette("pastel", len(self.area_features_crit_points))
        self.plot_crit_points(ax=ax)
        ax = plt.gca()
        self.plot(ax=ax, title="area features")
        areas = []
        for i, point in enumerate(self.area_features_crit_points):
            offset = min([self.data[point["start"]], self.data[point["end"]]])
            mask = (np.arange(self.n_samples) >= point["start"]) & (np.arange(self.n_samples) <= point["end"])
            a = ax.fill_between(
                self.time[mask],
                self.data[mask],
                offset,
                alpha=0.4,
                color=palette[i],
                edgecolor="black",
                lw=1,
                label=point["name"],
            )
            areas.append(a)
        leg = ax.legend(handles=areas, fontsize=self.fig_params["legend_size"])
        ax.add_artist(leg)

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20, ax=None):
        pass


class MultiChannelSignal:
    def __init__(self, signals: Dict[str, Signal]):
        self.signals = signals
        self.n_signals = len(signals)

        self.representations = {
            "whole_signal_waveforms": self.get_whole_signal_waveforms,
            "whole_signal_features": self.get_whole_signal_features,
            "windows_waveforms": self.get_windows_waveforms,
            "windows_features": self.get_windows_features,
        }

        signals_extraction_funcs = [{name: sig.extract_features for name, sig in self.signals.items()}]
        self.feature_extraction_funcs = OrderedDict(ChainMap(*signals_extraction_funcs))

    def __getitem__(self, key):
        return self.signals[key]

    def extract_features(self, return_arr=False, plot=False) -> Union[NDArray[Shape["FC"], Float], Dict[str, float]]:
        features = OrderedDict(
            {name: func(return_arr=False, plot=plot) for name, func in self.feature_extraction_funcs.items()}
        )
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
        return OrderedDict({name: sig.get_whole_signal_waveform() for name, sig in self.signals.items()})

    def get_whole_signal_features(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]:
        """Return whole signal features representation

        F - number of features from single signal
        C - number of channels

        If return_arr is True, then `NDArray[Shape["FC"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]`)
        """
        if return_arr:
            return self.extract_features(return_arr=True)
        return OrderedDict({name: func(return_arr=False) for name, func in self.feature_extraction_funcs.items()})

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
        return OrderedDict({name: sig.get_windows_waveforms(return_arr=False) for name, sig in self.signals.items()})

    def get_windows_features(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["B, FC"], Float], Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]:
        """Return windows features representation, i.e. features extracted for each window.

        B - number of windows
        F - number of features in a window
        C - number of channels

        If return_arr is True, then `NDArray[Shape["B, FC"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]`)
        """
        if return_arr:
            return np.concatenate(
                [sig.get_windows_features(return_arr=True) for _, sig in self.signals.items()], axis=1
            )
        return OrderedDict({name: sig.get_windows_features(return_arr=False) for name, sig in self.signals.items()})

    def plot(self, **kwargs):
        fig, axes = plt.subplots(self.n_signals, 1, figsize=(24, 3 * self.n_signals))
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot(ax=ax, **kwargs)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
        # TODO: Add windows plotting func


class MultiChannelPeriodicSignal(MultiChannelSignal):
    def __init__(self, signals: Dict[str, PeriodicSignal]):
        super().__init__(signals)

        self.representations.update(
            {
                "beats_waveforms": self.get_beats_waveforms,
                "beats_features": self.get_beats_features,
                "agg_beat_waveforms": self.get_agg_beat_waveforms,
                "agg_beat_features": self.get_agg_beat_features,
            }
        )

    def set_beats(self, source_channel=None, **kwargs):
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
            while not found_good_channel or len(source_channels) == 0:
                try:
                    signal = self.signals[new_source_channel]
                    signal.set_beats(**kwargs)
                    min_n_beats, max_n_beats = MIN_HR / 60 * signal.duration, MAX_HR / 60 * signal.duration
                    if min_n_beats <= len(signal.beats) <= max_n_beats:
                        found_good_channel = True
                        if new_source_channel != source_channel:
                            print(
                                f"Original source channel was corrupted. Found new source channel: {new_source_channel}"
                            )
                    else:
                        raise Exception("Number of beats indicates that HR bounds werent met. Trying another channel")
                except Exception as e:
                    print(e)
                    new_source_channel = source_channels.pop()
            if not found_good_channel:
                new_source_channel = source_channel
                self.signals[new_source_channel].set_beats(**kwargs)  # use Heart Rate (most dominant frequency)
            intervals = self.signals[new_source_channel]._get_beats_intervals()
        else:
            intervals = None
        for _, signal in self.signals.items():
            signal.set_beats(intervals, **kwargs)

    def set_agg_beat(self, **kwargs):
        for _, signal in self.signals.items():
            signal.set_agg_beat(**kwargs)

    def get_beats_features(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["B, FC"], Float], Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]:
        """Return beats features representation, i.e. features extracted for each beat.

        B - number of beats
        F - number of features in a beat
        C - number of channels

        If return_arr is True, then `NDArray[Shape["B, FC"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]`)
        """

        if return_arr:
            return np.concatenate([sig.get_beats_features(return_arr=True) for _, sig in self.signals.items()], axis=1)
        return OrderedDict({name: sig.get_beats_features(return_arr=False) for name, sig in self.signals.items()})

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
        return OrderedDict({name: sig.get_beats_waveforms(return_arr=False) for name, sig in self.signals.items()})

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
        return OrderedDict({name: sig.get_agg_beat_waveform() for name, sig in self.signals.items()})

    def get_agg_beat_features(
        self, return_arr=True, **kwargs
    ) -> Union[NDArray[Shape["F"], Float], Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]:
        """Return aggregated beat features representation, i.e. features extracted for each aggregated beat.

        F - number of features from single aggregated beat.
        C - number of channels

        If return_arr is True, then `NDArray[Shape["FC"], Float]` is returned
        If return_arr if False, then features for each channel are returned in dict form (`Dict[str, Dict[str, NDArray[Shape["F"], Float]]]]`)
        """
        if return_arr:
            return np.concatenate([sig.get_agg_beat_features(return_arr=True) for _, sig in self.signals.items()])
        return OrderedDict({name: sig.get_agg_beat_features(return_arr=False) for name, sig in self.signals.items()})

    def plot(self, **kwargs):
        fig, axes = plt.subplots(self.n_signals, 1, figsize=(24, 1 * self.n_signals), sharex=True)
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot(ax=ax, **kwargs)
            ax.set(xlabel="", ylabel="", title="")
            ax.grid(False)
            ax.get_legend().remove()
