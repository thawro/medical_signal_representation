from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

from signals.utils import (
    create_new_obj,
    get_windows,
    lazy_property,
    parse_nested_feats,
    z_score,
)


class BaseSignal:
    def __init__(self, name, data, fs, start_sec=0):
        self.name = name
        self.data = data
        self.fs = fs
        self.start_sec = start_sec
        self.n_samples = len(data)
        self.duration = self.n_samples / fs
        self.time = np.arange(0, self.n_samples) / fs
        self.end_sec = self.start_sec + self.time[-1]

    @lazy_property
    def troughs(self):
        troughs, _ = find_peaks(-self.data)
        return troughs

    @lazy_property
    def peaks(self):
        peaks, _ = find_peaks(self.data)
        return peaks

    def interpolate(self, fs, kind="cubic"):
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

    def resample(self, n_samples):
        new_data = resample(self.data, n_samples)
        fs_scaler = len(new_data) / self.n_samples
        new_fs = self.fs * fs_scaler
        new_name = self.name + f"_resampled({n_samples})"
        return create_new_obj(self, name=new_name, data=new_data, fs=new_fs)

    def resample_with_interpolation(self, n_samples, kind="pchip"):
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

    def z_score(self):
        new_data = z_score(self.data)
        new_name = self.name + "_z_scored"
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

    def min_max(self):
        arr = self.data
        new_data = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        new_name = self.name + "_min_max"
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

    def get_slice(self, start_time, end_time):
        mask = (self.time >= start_time) & (self.time < end_time)
        new_name = self.name + f"_slice({start_time}-{end_time})"
        new_data = self.data[mask]
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

    def is_in_time_bounds(self, start_s, end_s):
        return self.start_sec >= start_s and self.end_sec < end_s

    def get_derivative(self, deriv=1, window_length=None, polyorder=5):
        if window_length is None:
            window_length = int(np.ceil(5 / 100 * self.fs))
        if window_length % 2 == 0:
            window_length = window_length + 1
        new_name = self.name + f"_derivative({deriv})"
        new_data = savgol_filter(self.data, window_length, polyorder, deriv, delta=1 / self.fs)
        return BaseSignal(name=new_name, data=new_data, fs=self.fs, start_sec=self.start_sec)

    def detrend(self):
        detrended = detrend(self.data)
        new_name = self.name + f"_detrended"
        return create_new_obj(self, name=new_name, data=detrended, fs=self.fs)

    def get_cheby2_filtered(self, cutoff, btype="lowpass", order=4, rs=10):
        sos = cheby2(order, rs, cutoff, btype, output="sos", fs=self.fs)
        filtered = sosfiltfilt(sos, self.data)
        new_name = self.name + f"_{btype}_cheby2_filt({np.round(cutoff, 2)})"
        return create_new_obj(self, name=new_name, data=filtered, fs=self.fs)

    def get_fir_filtered(self, cutoff, btype="lowpass", numtaps=10):
        firf = firwin(numtaps, cutoff, pass_zero=btype, fs=self.fs)
        filtered = filtfilt(firf, 1, self.data)
        new_name = self.name + f"{btype}_fir_filt({np.round(cutoff, 2)})"
        return create_new_obj(self, name=new_name, data=filtered, fs=self.fs)

    def get_notch_filtered(self, cutoff, Q=10):
        b, a = iirnotch(cutoff, Q=Q, fs=self.fs)
        filtered = filtfilt(b, a, self.data)
        new_name = self.name + f"_notch_filt({np.round(cutoff, 2)})"
        return create_new_obj(self, name=new_name, data=filtered, fs=self.fs)

    def seasonal_decompose(self, plot=False):
        res = STL(self.data, period=self.fs).fit()
        if plot:
            res.plot()
        return res

    def fft(self, plot=False, ax=None):
        mags = rfft(self.data)
        freqs = rfftfreq(self.n_samples, 1 / self.fs)
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(freqs[1:], np.abs(mags[1:]))
        return freqs, mags

    def psd(self, window_size, min_hz=0, max_hz=10, plot=False, ax=None):
        freqs, pxx = welch(self.data, self.fs, nperseg=self.fs * window_size)
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(16, 8))
            mask = (freqs >= min_hz) & (freqs <= max_hz)
            ax.plot(freqs[mask], pxx[mask], lw=2)
            ax.set_xlabel("Frequency [Hz]", fontsize=18)
            ax.set_ylabel("Power Spectral Density", fontsize=18)

        return freqs, pxx

    def spectrogram(self, NFFT=256, noverlap=128, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 8))
        spectrum, freqs, t, img = ax.specgram(self.data, Fs=self.fs, NFFT=NFFT, noverlap=noverlap, cmap="viridis")
        cbar = plt.gcf().colorbar(img, ax=ax)
        cbar.ax.set_title("Intensity", fontsize=14)
        ax.set_ylabel("Frequency [Hz]", fontsize=18)
        ax.set_xlabel("Time [s]", fontsize=18)
        return spectrum, freqs, t

    def extract_basic_features(self, return_arr=True):
        features = OrderedDict(
            {
                "mean": np.mean(self.data),
                "std": np.std(self.data),
                "median": np.median(self.data),
                "skewness": skew(self.data),
                "kurtosis": kurtosis(self.data),
            }
        )
        if return_arr:
            return np.array(list(features.values()))
        return features

    def extract_features(self, return_arr=True):
        features = OrderedDict({"basic_features": self.extract_basic_features(return_arr=return_arr)})
        if return_arr:
            return np.array(list(parse_nested_feats(features).values()))
        return features

    def extract_embedding(self, embedding_model):
        embedding = embedding_model(self.data)
        return embedding

    def plot(
        self, start_time=0, width=10, scatter=False, line=True, first_der=False, label=None, use_samples=False, ax=None
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(24, 6))
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
        ax.set_title(self.name, fontsize=22)

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
        intervals = get_windows(0, self.n_samples, win_len_samples, step_samples)
        intervals = np.array(intervals).astype(int)
        self.windows = [
            create_new_obj(self, data=self.data[start:end], start_sec=self.time[start]) for start, end in intervals
        ]

    def plot_windows(self, **kwargs):
        for window in self.windows:
            window.plot(**kwargs)

    def get_whole_signal_waveform(self):
        return self.data

    def get_windows_waveforms(self):
        return np.array([window.data for window in self.windows])

    def get_windows_features(self, return_arr=True):
        if return_arr:
            return np.array([window.extract_features(return_arr=True) for window in self.windows])
        return OrderedDict(
            {f"window_{i}": window.extract_features(return_arr=False) for i, window in enumerate(self.windows)}
        )


class PeriodicSignal(ABC, Signal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.valid_beats_mask = None
        self.cleaned = np.copy(self.data)

    @abstractmethod
    def set_beats(self, resample=True, n_samples=100, validate=True, plot=False, use_raw=True):
        pass

    def set_windows(self, win_len_s, step_s):
        # aggregate beats for each window
        super().set_windows(win_len_s, step_s)
        for window in self.windows:
            window.beats = np.array(
                [beat for beat in self.beats if beat.is_in_time_bounds(window.start_sec, window.end_sec)]
            )
            window.aggregate()

    def set_agg_beat(self, valid_only=True):
        # agreguje listę uderzeń w pojedyncze uderzenie sPPG i je zwraca (jako obiekt PPGBeat)
        beats_to_aggregate_mask = self.valid_beats_mask if valid_only else len(self.beats) * [True]
        if beats_to_aggregate_mask is None:
            beats_to_aggregate_mask = len(self.beats) * [True]
        beats_to_aggregate = self.beats[beats_to_aggregate_mask]
        beats_data = np.array([beat.data for beat in beats_to_aggregate])
        beats_times = np.array([beat.time for beat in beats_to_aggregate])
        agg_beat_data, agg_beat_time = beats_data.mean(axis=0), beats_times.mean(axis=0)
        agg_fs = len(agg_beat_data) / (agg_beat_time[-1] - agg_beat_time[0])
        BeatClass = beats_to_aggregate[0].__class__
        self.agg_beat = BeatClass("agg_beat", agg_beat_data, agg_fs, start_sec=0)

    def get_beats_features(self, return_arr=True):
        if return_arr:
            return np.array([beat.extract_features(return_arr=True) for beat in self.beats])
        return OrderedDict({f"beat_{beat.beat_num}": beat.extract_features(return_arr=False) for beat in self.beats})

    def get_beats_waveforms(self):
        return np.array([beat.data for beat in self.beats])

    def plot_beats_segmentation(self, use_raw=False, axes=None):
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(24, 4), gridspec_kw={"width_ratios": [8, 2]})
        if use_raw:
            self.plot(ax=axes[0])
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

    def plot_beats(self, same_ax=True, ax=None, plot_valid=True, plot_invalid=True):
        beats_to_plot = [
            beat for beat in self.beats if (beat.is_valid and plot_valid) or (not beat.is_valid and plot_invalid)
        ]

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

    @abstractmethod
    def extract_crit_points_features(self, plot=True):
        pass

    @lazy_property
    def crit_points(self):
        return {}

    def plot_crit_points(self, points=None, plot_sig=False, use_samples=False, offset=0, ax=None, **kwargs):
        if points is None:
            points = list(self.crit_points.keys())
        xaxes_data = np.arange(self.n_samples) if use_samples else self.time
        xrange = xaxes_data[-1] - xaxes_data[0]
        yrange = self.data.max() - self.data.min()

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))

        if plot_sig:
            self.plot(ax=ax)
        for name, loc in self.crit_points.items():
            if name not in points:
                continue
            t = xaxes_data[loc]
            val = self.data[loc] + offset
            ax.scatter(t, val, label=name, s=160)
            ax.annotate(name.upper(), (t + 0.012 * xrange, val + 0.012 * yrange), fontweight="bold", fontsize=18)
        # plt.legend()

    def plot(
        self,
        start_time=0,
        width=10,
        scatter=False,
        line=True,
        first_der=False,
        label=None,
        use_samples=False,
        with_crit_points=True,
        ax=None,
    ):
        super().plot(
            start_time=start_time,
            width=width,
            scatter=scatter,
            line=line,
            first_der=first_der,
            label=label,
            use_samples=use_samples,
            ax=ax,
        )
        ax = plt.gca()
        if with_crit_points:
            self.plot_crit_points(use_samples=use_samples, ax=ax)

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20, ax=None):
        pass


class MultiChannelSignal:
    def __init__(self, signals: Dict[str, PeriodicSignal]):
        self.signals = signals
        self.n_signals = len(signals)

        self.representations = {
            "whole_signal_waveforms": self.get_whole_signal_waveforms,
            "whole_signal_features": self.get_whole_signal_features,
            "windows_waveforms": self.get_windows_waveforms,
            "windows_features": self.get_windows_features,
        }

    def set_windows(self, win_len_s, step_s, **kwargs):
        self.windows = {name: sig.get_windows(win_len_s, step_s) for name, sig in self.signals.items()}

    def get_whole_signal_waveforms(self, **kwargs):
        return np.array([sig.data for name, sig in self.signals.items()])

    def get_whole_signal_features(self, return_arr=True, **kwargs):
        if return_arr:
            return np.array([sig.extract_features(return_arr=True) for _, sig in self.signals.items()])
        return OrderedDict({name: sig.extract_features(return_arr=False) for name, sig in self.signals.items()})

    def get_windows_waveforms(self, **kwargs):
        return np.array([sig.extract_per_window_waveforms() for _, sig in self.signals.items()])

    def get_windows_features(self, return_arr=True, **kwargs):
        if return_arr:
            return np.array([sig.extract_per_window_features(return_arr=True) for _, sig in self.signals.items()])
        return OrderedDict(
            {name: sig.extract_per_window_features(return_arr=False) for name, sig in self.signals.items()}
        )

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

    def set_beats(self, **kwargs):
        for _, signal in self.signals.items():
            signal.set_beats(**kwargs)
        self.beats = {name: signal.beats for name, signal in self.signals.items()}

    def set_agg_beat(self):
        for _, signal in self.signals.items():
            signal.set_agg_beat(**kwargs)
        self.agg_beat = {name: signal.agg_beat for name, signal in self.signals.items()}

    def get_beats_features(self, return_arr=True, **kwargs):
        if return_arr:
            return np.array([sig.get_beats_features(return_arr=True) for _, sig in self.signals.items()])
        return OrderedDict({name: sig.get_beats_features(return_arr=False) for name, sig in self.signals.items()})

    def get_beats_waveforms(self, **kwargs):
        return np.array([sig.get_beats_waveforms() for _, sig in self.signals.items()])

    def get_agg_beat_waveforms(self, **kwargs):
        return np.array([signal.agg_beat.data for _, signal in self.signals.items()])

    def get_agg_beat_features(self, return_arr=True, **kwargs):
        agg_beats = OrderedDict({name: signal.agg_beat for name, signal in self.signals.items()})
        if return_arr:
            return np.array([agg_beat.extract_features(return_arr=True) for _, agg_beat in agg_beats.items()])
        return OrderedDict({name: agg_beat.extract_features(return_arr=False) for name, agg_beat in agg_beats.items()})

    def plot(self, **kwargs):
        fig, axes = plt.subplots(self.n_signals, 1, figsize=(24, 1 * self.n_signals), sharex=True)
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot(ax=ax, **kwargs)
            ax.set(xlabel="", ylabel="", title="")
            ax.grid(False)
            ax.get_legend().remove()
