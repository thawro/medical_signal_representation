from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.signal import (
    butter,
    cheby2,
    detrend,
    filtfilt,
    find_peaks,
    firwin,
    freqz,
    iirnotch,
    resample,
    savgol_filter,
    sosfiltfilt,
    spectrogram,
    welch,
)
from scipy.stats import kurtosis, skew
from statsmodels.tsa.seasonal import STL

from signals.utils import create_new_obj, lazy_property, parse_nested_feats, z_score


class BaseSignal:
    def __init__(self, name, data, fs, start_sec=0, end_sec=None):
        self.name = name
        self.data = data
        self.fs = fs
        self.start_sec = start_sec
        self.end_sec = end_sec
        self.n_samples = len(data)
        self.duration = self.n_samples / fs
        self.time = np.arange(0, self.n_samples) / fs

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

    def get_slice(self, start_time, end_time):
        mask = (self.time >= start_time) & (self.time <= end_time)
        new_name = self.name + f"_slice({start_time}-{end_time})"
        new_data = self.data[mask]
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

    def get_derivative(self, deriv=1, smoothing=1, polyorder=5):
        noise_factor = (4 + 4 * deriv + 2 * polyorder) * self.fs / 100
        window_length = int(noise_factor * smoothing) + 2
        if window_length % 2 == 0:
            window_length = window_length + 1
        new_name = self.name + f"_derivative({deriv})"
        new_data = savgol_filter(self.data, window_length, polyorder, deriv, delta=1 / self.fs)
        return BaseSignal(name=new_name, data=new_data, fs=self.fs, start_sec=self.start_sec, end_sec=self.end_sec)

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


class PeriodicSignal(ABC, BaseSignal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.beats = None  # List of PPGBeat objects
        self.valid_beats_mask = None
        self.agg_beat = None  # PPGBeat object
        self.agg_beat_features = None
        self.cleaned = np.copy(self.data)

    @abstractmethod
    def get_beats(self, resample=True, n_samples=100, validate=True, plot=False, use_raw=True, return_arr=False):
        pass

    def plot_beats_segmentation(self, use_raw=False, ax=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(24, 3))
        if use_raw:
            self.plot(ax=ax)
        else:
            ax.plot(self.time, self.cleaned, lw=3, label=self.name)
        beats_bounds = []
        for beat in self.beats:
            bounds = [beat.start_sec, beat.end_sec]
            color = "green" if beat.is_valid else "red"
            ax.fill_between(bounds, self.data.min(), self.data.max(), alpha=0.15, color=color)
            beats_bounds.extend(bounds)
        ax.vlines(beats_bounds, self.data.min(), self.data.max(), lw=1.5, ec="black", ls="--")
        ax.grid(False)
        ax.set_ylim([self.data.min(), self.data.max()])
        ax.legend()

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
            ax.legend()
        else:
            ncols = 5
            nrows = int(np.ceil(n_beats / ncols))

            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            for beat, ax in zip(beats_to_plot, axes.flatten()):
                palette = valid_palette if beat.is_valid else invalid_palette
                ax.plot(beat.time, beat.data, lw=3, c=palette[int(n_beats / 1.5)])
                ax.set_title(beat.name, fontsize=18)
            plt.tight_layout()

    def aggregate(self, valid_only=True, **kwargs):
        # agreguje listę uderzeń w pojedyncze uderzenie sPPG i je zwraca (jako obiekt PPGBeat)
        if self.beats is None:
            self.get_beats(**kwargs)
        beats_to_aggregate_mask = self.valid_beats_mask if valid_only else len(self.beats) * [True]
        if beats_to_aggregate_mask is None:
            beats_to_aggregate_mask = len(self.beats) * [True]
        beats_to_aggregate = self.beats[beats_to_aggregate_mask]
        beats_data = np.array([beat.data for beat in beats_to_aggregate])
        beats_times = np.array([beat.time for beat in beats_to_aggregate])
        agg_beat_data, agg_beat_time = beats_data.mean(axis=0), beats_times.mean(axis=0)
        agg_fs = len(agg_beat_data) / (agg_beat_time[-1] - agg_beat_time[0])
        BeatClass = beats_to_aggregate[0].__class__
        agg_beat = BeatClass("agg_beat", agg_beat_data, agg_fs, start_sec=0, end_sec=agg_beat_time[-1])
        self.agg_beat = agg_beat
        return self.agg_beat

    def extract_per_beat_features(self, plot=False, return_arr=True, n_beats=-1):
        n_beats = len(self.beats) if n_beats <= 0 else n_beats
        beats = self.beats[:n_beats]
        if return_arr:
            return np.array([beat.extract_features(plot=plot, return_arr=True) for beat in beats])
        return OrderedDict(
            {f"beat_{beat.beat_num}": beat.extract_features(plot=plot, return_arr=False) for beat in beats}
        )

    def extract_per_beat_waveforms(self, return_arr=True, n_beats=-1):
        n_beats = len(self.beats) if n_beats <= 0 else n_beats
        beats = self.beats[:n_beats]
        if return_arr:
            return np.array([beat.data for beat in beats])
        return OrderedDict({f"beat_{beat.beat_num}": beat.data for beat in beats})

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20):
        super().explore(start_time, width, window_size, min_hz, max_hz)
        self.plot_beats_segmentation()
        fig, axes = plt.subplots(1, 2, figsize=(24, 5))
        self.plot_beats(ax=axes[0])
        self.agg_beat.plot(with_crit_points=True, ax=axes[1])


class BeatSignal(ABC, BaseSignal):
    def __init__(self, name, data, fs, start_sec, end_sec=None, beat_num=0, is_valid=True):
        super().__init__(name, data, fs, start_sec, end_sec)
        self.is_valid = is_valid
        self.beat_num = beat_num

    @abstractmethod
    def extract_crit_points_features(self, plot=True):
        pass

    @lazy_property
    def crit_points(self):
        return {}

    def plot_crit_points(self, use_samples=False, offset=0, ax=None, **kwargs):
        xaxes_data = np.arange(self.n_samples) if use_samples else self.time
        xrange = xaxes_data[-1] - xaxes_data[0]
        yrange = self.data.max() - self.data.min()

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7))

        for name, loc in self.crit_points.items():
            t = xaxes_data[loc]
            val = self.data[loc] + offset
            ax.scatter(t, val, label=name, s=160)
            ax.annotate(name.upper(), (t + 0.012 * xrange, val + 0.012 * yrange), fontweight="bold", fontsize=18)
        plt.legend()

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

    def plot(self, **kwargs):
        fig, axes = plt.subplots(self.n_signals, 1, figsize=(24, 3 * self.n_signals))
        for ax, (sig_name, sig) in zip(axes, self.signals.items()):
            sig.plot(ax=ax, **kwargs)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")


class MultiChannelPeriodicSignal(MultiChannelSignal):
    def __init__(self, signals: Dict[str, PeriodicSignal]):
        super().__init__(signals)
        self.beats = None
        self.agg_beats = None

    def get_beats(self, **kwargs):
        self.beats = {name: signal.get_beats(**kwargs) for name, signal in self.signals.items()}
        return self.beats

    def aggregate(self, **kwargs):
        self.agg_beat = {name: signal.aggregate(**kwargs) for name, signal in self.signals.items()}
        return self.agg_beat
