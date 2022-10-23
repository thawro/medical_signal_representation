from typing import Type

from signals.base import MultiChannelSignal


class RepresentationExtractor:
    def __init__(measurement: Type[MultiChannelSignal]):
        self.measurement = measurement

    def set_windows(self, win_len, step):
        self.measurement.set_windows(win_len, step)

    def get_representations(self, **kwargs):
        return {rep_name: rep_func(**kwargs) for rep_name, rep_func in self.measurement.representations.items()}


class PeriodicRepresentationExtractor(RepresentationExtractor):
    def set_beats(self, **kwargs):
        self.measurement.set_beats(**kwargs)

    def set_agg_beat(self, **kwargs):
        self.measurement.set_agg_beat(**kwargs)

    def get_representations(self, n_beats=-1, **kwargs):
        representations = {
            rep_name: rep_func(**kwargs) for rep_name, rep_func in self.measurement.representations.items()
        }
        actual_n_beats = representations["beats_waveforms"].shape[1]
        if n_beats == -1:
            n_beats = actual_n_beats
        if n_beats < actual_n_beats:
            representations["beats_waveforms"] = representations["beats_waveforms"][:, :n_beats]
            representations["beats_features"] = representations["beats_features"][:, :n_beats]
        elif n_beats > actual_n_beats:
            n_more_beats = n_beats - actual_n_beats
            pad_width = ((0, 0), (0, n_more_beats), (0, 0))
            representations["beats_waveforms"] = np.pad(representations["beats_waveforms"], pad_width)
            representations["beats_features"] = np.pad(representations["beats_features"], pad_width)
        return representations
