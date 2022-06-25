from signals.base import PeriodicSignal, BeatSignal
from biosppy.signals.ecg import ecg
import numpy as np

class ECGSignal(PeriodicSignal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.ecg_info = ecg(data, fs, show=False)
        self.rpeaks = None
        
    def find_rpeaks(self):
        self.rpeaks = self.ecg_info['rpeaks']
        return self.rpeaks
    
    
    def get_beats(self, resample_beats=True, n_samples=100, validate=True, plot=False):
        if self.rpeaks is None:
            self.rpeaks()
        beats = self.ecg_info['templates']
        beats_time = self.ecg_info['templates_ts']
        beats_fs = len(beats_time) / beats_time[-1]
        self.beats = []
        for i, beat in enumerate(beats):
            self.beats.append(ECGBeat(f"beat_{i}", beat, beats_fs, start_sec=i*beats_fs))
        self.beats = np.array(self.beats)
        return self.beats
    
    
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
    
    
    def extract_ecg_features(self):
        if self.rpeaks is None:
            self.find_rpeaks()
        r_peaks = self.rpeaks
        r_vals = self.data[r_peaks]
        r_times = self.time[r_peaks]
        ibi = np.diff(r_times)       
    
        self.ecg_features = {
            'ibi_mean': np.mean(ibi),
            'ibi_std': np.std(ibi),
            'R_val': np.mean(r_vals)
        }
        
        return self.ecg_features
    
    def extract_features(self, plot=True):
        self.features = super().extract_features()
        ecg_features = {
            "ecg_features": self.extract_ecg_features()
        }
        self.features.update(ecg_features) 
        return self.features
    
    
class ECGBeat(BeatSignal):
    def __init__(self, name, data, fs, start_sec, beat_num=0):
        super().__init__(name, data, fs, start_sec, beat_num)
        
    def extract_agg_beat_features(self, plot=True):
        pass