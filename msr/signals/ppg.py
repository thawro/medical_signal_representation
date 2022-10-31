import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import integrate
from scipy.signal import butter, filtfilt, find_peaks

from msr.signals.base import BeatSignal, PeriodicSignal
from msr.signals.utils import FIGSIZE_2, get_outliers_mask, parse_nested_feats, z_score


def get_no_outlier_beats_mask(beats, IQR_scale=1.5):
    beats_data = np.array([beat.data for beat in beats])
    beats_times = np.array([beat.time for beat in beats])

    mean_beats = beats_data.mean(axis=1)
    beats_durations = np.array([t[-1] - t[0] for t in beats_times])
    min_vals = np.array([v.min() for v in beats_data])
    max_vals = np.array([v.max() for v in beats_data])

    data_mask = get_outliers_mask(mean_beats, IQR_scale=IQR_scale)
    duration_mask = get_outliers_mask(beats_durations, IQR_scale=IQR_scale)
    min_vals_mask = get_outliers_mask(min_vals, IQR_scale=IQR_scale)
    max_vals_mask = get_outliers_mask(max_vals, IQR_scale=IQR_scale)

    valid_beats_mask = data_mask & duration_mask & min_vals_mask & max_vals_mask
    return valid_beats_mask


def get_valid_beats_mask(beats, max_duration=1.1):
    duration_mask = np.array([beat.duration <= max_duration for beat in beats])
    no_outlier_beats_mask = get_no_outlier_beats_mask(beats)
    return duration_mask & no_outlier_beats_mask


def moving_average(data, winsize):
    extended_data = np.concatenate((np.ones(winsize) * data[0], data, np.ones(winsize) * data[-1]))
    win = np.ones(winsize) / winsize
    smoothed = np.convolve(extended_data, win, mode="same")
    return smoothed[winsize:-winsize]


def find_systolic_peaks_ELGENDI(data, fs, f1=0.5, f2=8, w1=0.111, w2=0.667, beta=0.2):
    cutoff = [f1, f2]
    cutoff = 2.0 * np.array(cutoff) / fs
    b, a = butter(N=2, Wn=cutoff, btype="bandpass", analog=False, output="ba")
    filtered = filtfilt(b, a, data)  # 2
    clipped = np.copy(filtered)  # 3
    clipped[clipped < 0] = 0
    squared = clipped**2  # 4

    ma_peak = moving_average(squared, int(w1 * fs))  # 5
    ma_beat = moving_average(squared, int(w2 * fs))  # 6

    z_mean = squared.mean()  # 7
    alpha = beta * z_mean  # 8

    thr_1 = ma_beat + alpha  # 9

    #  10 - 16
    valid_segments = ma_peak > thr_1
    starts = np.where(~valid_segments[0:-1] & valid_segments[1:])[0]
    ends = np.where(valid_segments[0:-1] & ~valid_segments[1:])[0]
    ends = ends[ends > starts[0]]

    blocks = zip(starts, ends)  # 17
    thr_2 = w1  # 18

    peaks = []
    #  19 - 25
    for start, end in blocks:
        if end - start >= thr_2:
            peaks.append(data[start:end].argmax() + start)
    return np.array(peaks)


class PPGSignal(PeriodicSignal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)

    def find_troughs(self):
        data_z_score = z_score(self.data)
        troughs, _ = find_peaks(-data_z_score, height=0.5)
        self.troughs = troughs
        return troughs

    def find_peaks(self):
        peaks_idxs = find_systolic_peaks_ELGENDI(self.data, self.fs)
        self.peaks = peaks_idxs
        return self.peaks

    def set_beats(self, resample_beats=True, n_samples=100, validate=True, plot=False):
        if self.peaks is None:
            self.find_peaks()
        if self.troughs is None:
            self.find_troughs()

        peaks_time = self.time[self.peaks]
        peaks_values = self.data[self.peaks]

        beats = []
        for i, (trough_start, trough_end) in enumerate(zip(self.troughs[:-1], self.troughs[1:])):
            peaks_between = self.peaks[(self.peaks > trough_start) & (self.peaks < trough_end)]
            if len(peaks_between) == 0:
                continue
            beat_data = self.data[trough_start:trough_end]
            beat_time = (
                self.time[trough_start:trough_end] - self.time[trough_start]
            )  # removing first time value to get proper beat timing
            beat = PPGBeat(f"beat_{i}", beat_data, fs=self.fs, start_sec=self.time[trough_start], beat_num=len(beats))
            if resample_beats:
                beat = beat.resample(n_samples)
            beats.append(beat)

        self.beats = np.array(beats, dtype=PPGBeat)
        self.valid_beats_mask = get_valid_beats_mask(self.beats)
        for beat, is_valid in zip(self.beats, self.valid_beats_mask):
            beat.is_valid = is_valid

        if plot:
            # self.plot(extrema=True)
            fig, axes = plt.subplots(1, 2, figsize=(24, 5))
            self.aggregate(valid_only=False)
            self.plot_beats(plot_valid=True, plot_invalid=True, ax=axes[0])
            self.aggregate(valid_only=True)
            self.plot_beats(plot_valid=True, plot_invalid=False, ax=axes[1])
        else:
            self.aggregate(valid_only=True)

    def extract_hrv_features(self):
        if self.hrv_features is None:
            self.find_peaks()  # hrv_features are calculated there
        return self.hrv_features

    def extract_features(self, plot=True, parse=True):
        # rozwinięcie cech podstawowej klasy Signal o cechy typowe dla sygnału PPG (do wyczytania z artykułów)
        if self.agg_beat is None:
            self.get_beats(plot=plot)
        self.features = super().extract_features()
        ppg_features = {
            "agg_beat_features": self.agg_beat.extract_features(plot=plot),
            "hrv_features": self.extract_hrv_features(),
        }
        self.features.update(ppg_features)
        features = parse_nested_feats(self.features) if parse else self.features
        features = {f"{self.name}__{feat_name}": feat_val for feat_name, feat_val in features.items()}
        return features

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20):
        super().explore(start_time, width, window_size, min_hz, max_hz)


class PPGBeat(BeatSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0):
        super().__init__(name, data, fs, start_sec, beat_num)
        self.find_systolic_peak()

    def find_systolic_peak(self):
        # metoda znajdująca pik systolic (algorytm do znalezienia w artykułach)
        self.systolic_peak_loc = self.data.argmax()  # XD
        self.systolic_peak_time = self.time[self.systolic_peak_loc]
        self.systolic_peak_val = self.data[self.systolic_peak_loc]

    def find_dicrotic_notch(self):
        # metoda znajdująca dołek dicrotic (algorytm do znalezienia w artykułach)
        pass

    def find_diastolic_peak(self):
        # metoda znajdująca pik diastolic (algorytm do znalezienia w artykułach)
        pass

    def extract_agg_beat_features(self, plot=True):
        systolic_onset_slope = (self.systolic_peak_val - self.data[0]) / self.systolic_peak_time
        systolic_offset_slope = (self.systolic_peak_val - self.data[-1]) / (self.duration - self.systolic_peak_time)
        energy = (self.data**2).mean()
        before_systolic_area = integrate.simpson(
            abs(self.data[: self.systolic_peak_loc]), self.time[: self.systolic_peak_loc]
        )
        after_systolic_area = integrate.simpson(
            abs(self.data[self.systolic_peak_loc :]), self.time[self.systolic_peak_loc :]
        )

        if plot:
            fig, ax = plt.subplots(figsize=FIGSIZE_2)
            ax.plot(self.time, self.data, c=sns.color_palette()[0], lw=3)
            ax.scatter(
                self.time[self.systolic_peak_loc],
                self.data[self.systolic_peak_loc],
                s=200,
                c="r",
                marker="^",
                label="systolic peak",
            )
            ax.fill_between(
                self.time,
                self.data,
                self.data.min(),
                where=self.time <= self.time[self.systolic_peak_loc],
                color="g",
                alpha=0.2,
            )
            ax.fill_between(
                self.time,
                self.data,
                self.data.min(),
                where=self.time >= self.time[self.systolic_peak_loc],
                color="r",
                alpha=0.2,
            )
            ax.set_title(f"Pojedyncze uderzenie PPG", fontsize=24)
            ax.set_xlabel("Czas [s]", fontsize=20)
            ax.set_ylabel("j.u.", fontsize=20)
            ax.legend()
            fig.savefig("sppg.png", dpi=500)

        self.agg_beat_features = {
            "duration": self.duration,
            "systolic_peak_val": self.systolic_peak_val,
            "systolic_peak_time": self.systolic_peak_time,
            "systolic_onset_slope": systolic_onset_slope,
            "systolic_offset_slope": systolic_offset_slope,
            "energy": energy,
            "before_systolic_area": before_systolic_area,
            "after_systolic_area": after_systolic_area,
        }
        return self.agg_beat_features

    def extract_features(self, plot=True):
        # rozwinięcie cech podstawowej klasy Signal o cechy typowe dla zagregowanego sygnału PPG, czyli sPPG (do wyczytania z artykułów)
        self.features = super().extract_features()
        sppg_features = self.extract_agg_beat_features(plot=plot)
        self.features.update(sppg_features)
        return self.features

    def explore(self, start_time=0, width=None, window_size=4, min_hz=0, max_hz=20):
        super().explore(start_time, width, window_size, min_hz, max_hz)
