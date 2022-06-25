from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
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

from signals.utils import create_new_obj, z_score


class BaseSignal:
    def __init__(self, name, data, fs, start_sec=0):
        self.name = name
        self.data = data
        self.fs = fs
        self.start_sec = start_sec
        self.n_samples = len(data)
        self.duration = self.n_samples / fs
        self.time = np.arange(0, self.n_samples) / fs
        self.peaks = None
        self.troughs = None

    def find_troughs(self):
        self.troughs, _ = find_peaks(-self.data)
        return self.troughs

    def find_peaks(self):
        self.peaks, _ = find_peaks(self.data)
        return self.peaks

    def interpolate(self, fs, kind="cubic"):
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

    def z_score(self):
        new_data = z_score(self.data)
        new_name = self.name + "_z_scored"
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

    def get_slice(self, start_time, end_time):
        mask = (self.time >= start_time) & (self.time <= end_time)
        new_name = self.name + f"_slice({start_time}-{end_time})"
        new_data = self.data[mask]
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

    def get_derivative(self, deriv=1, window_length=21, polyorder=5):
        new_name = self.name + f"_derivative({deriv})"
        new_data = savgol_filter(self.data, window_length, polyorder, deriv, delta=1 / self.fs)
        return create_new_obj(self, name=new_name, data=new_data, fs=self.fs)

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

    def extract_basic_features(self):
        self.basic_features = {
            "mean": np.mean(self.data),
            "std": np.std(self.data),
            "median": np.median(self.data),
            "skewness": skew(self.data),
            "kurtosis": kurtosis(self.data),
        }
        return self.basic_features

    def extract_features(self):
        self.features = {"basic_features": self.extract_basic_features()}
        return self.features

    def plot(self, start_time=0, width=10, scatter=False, line=True, first_der=False, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(18, 6))
        if width == -1:
            width = self.time[-1] - start_time
        end_time = start_time + width
        signal_slice = self.get_slice(start_time, end_time)
        if scatter:
            sig_plot = ax.scatter(signal_slice.time, signal_slice.data, lw=3, label=signal_slice.name)
            plots = [sig_plot]
        if line:
            sig_plot = ax.plot(signal_slice.time, signal_slice.data, lw=3, label=signal_slice.name)
            plots = [sig_plot[0]]
        if first_der:
            ax2 = ax.twinx()
            signal_slice_der = signal_slice.get_derivative()
            if scatter:
                first_der_plot = ax2.scatter(
                    signal_slice_der.time,
                    signal_slice_der.data,
                    c=sns.color_palette()[1],
                    lw=1,
                    label=signal_slice_der.name,
                )
            if line:
                first_der_plot = ax2.plot(
                    signal_slice_der.time,
                    signal_slice_der.data,
                    c=sns.color_palette()[1],
                    lw=1,
                    label=signal_slice_der.name,
                )
            plots.append(first_der_plot[0])
            ax2.set_ylabel("First derivative values", fontsize=18)
            ax2.axhline(y=0, c="black", ls="--", lw=0.5)
        labels = [plot.get_label() for plot in plots]
        ax.legend(plots, labels, loc=0, fontsize=18)
        ax.set_ylabel("values", fontsize=18)
        ax.set_xlabel("Time [s]", fontsize=18)
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

    @abstractmethod
    def get_beats(self, resample_beats=True, n_samples=100, validate=True, plot=False):
        pass

    def plot_beats(self, same_ax=True, ax=None, plot_valid=True, plot_invalid=True):
        if self.beats is None:
            self.get_beats()
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
                fig, ax = plt.subplots(figsize=FIGSIZE_2)
            for i, beat in enumerate(beats_to_plot):
                palette = valid_palette if beat.is_valid else invalid_palette
                ax.plot(beat.time, beat.data, color=palette[i])
            ax.plot(self.agg_beat.time, self.agg_beat.data, lw=8, alpha=0.5, c="black", label="Aggregated beat")
            ax.set_title(f"{n_beats} beats ({n_valid} valid, {n_invalid} invalid)", fontsize=22)
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
        beats_to_aggregate = self.beats[beats_to_aggregate_mask]
        beats_data = np.array([beat.data for beat in beats_to_aggregate])
        beats_times = np.array([beat.time for beat in beats_to_aggregate])
        agg_beat_data, agg_beat_time = beats_data.mean(axis=0), beats_times.mean(axis=0)
        agg_fs = len(agg_beat_data) / (agg_beat_time[-1] - agg_beat_time[0])
        BeatClass = beats_to_aggregate[0].__class__
        agg_beat = BeatClass("agg_beat", agg_beat_data, agg_fs, start_sec=0)
        self.agg_beat = agg_beat
        return self.agg_beat

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20):
        super().explore(start_time, width, window_size, min_hz, max_hz)
        self.get_beats(plot=True)
        self.agg_beat.explore(start_time=0)


class BeatSignal(ABC, BaseSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0, is_valid=True):
        super().__init__(name, data, fs, start_sec)
        self.is_valid = is_valid
        self.beat_num = beat_num

    @abstractmethod
    def extract_agg_beat_features(self, plot=True):
        pass

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20):
        self.extract_agg_beat_features(plot=True)
