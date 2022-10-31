from typing import List, Type

import numpy as np

from msr.signals.base import MultiChannelSignal


class RepresentationExtractor:
    def __init__(self, measurement: Type[MultiChannelSignal]):
        self.measurement = measurement

    def set_windows(self, win_len_s, step_s):
        self.measurement.set_windows(win_len_s, step_s)

    def get_representations(self, representation_types: List[str], **kwargs):
        return {rep_type: self.measurement.representations[rep_type](**kwargs) for rep_type in representation_types}


class PeriodicRepresentationExtractor(RepresentationExtractor):
    def set_beats(self, **kwargs):
        self.measurement.set_beats(**kwargs)

    def set_agg_beat(self, **kwargs):
        self.measurement.set_agg_beat(**kwargs)

    def get_representations(self, representation_types, n_beats=-1, **kwargs):
        representations = super().get_representations(representation_types, **kwargs)
        for beats_rep_name in ["beats_waveforms", "beats_features"]:
            if beats_rep_name in representation_types:
                actual_n_beats = representations[beats_rep_name].shape[1]
                if n_beats == -1:
                    n_beats = actual_n_beats
                if n_beats < actual_n_beats:
                    representations[beats_rep_name] = representations[beats_rep_name][:, :n_beats]
                elif n_beats > actual_n_beats:
                    n_more_beats = n_beats - actual_n_beats
                    pad_width = ((0, 0), (0, n_more_beats), (0, 0))
                    representations[beats_rep_name] = np.pad(representations[beats_rep_name], pad_width)
        return representations
