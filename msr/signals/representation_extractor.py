from typing import List, Type

import numpy as np

from msr.signals.base import MultiChannelSignal


class RepresentationExtractor:
    def __init__(self, measurement: Type[MultiChannelSignal]):
        self.measurement = measurement

    def set_windows(self, win_len_s, step_s):
        self.measurement.set_windows(win_len_s, step_s)

    def get_representations(self, representation_types: List[str], return_arr: bool = True, **kwargs):
        if representation_types == "all":
            representation_types = list(self.measurement.representations.keys())
        return {
            rep_type: self.measurement.representations[rep_type](return_arr=return_arr, **kwargs)
            for rep_type in representation_types
        }


class PeriodicRepresentationExtractor(RepresentationExtractor):
    def set_beats(self, **kwargs):
        self.measurement.set_beats(**kwargs)

    def set_agg_beat(self, **kwargs):
        self.measurement.set_agg_beat(**kwargs)

    def get_representations(
        self, representation_types: List[str] = "all", return_arr: bool = True, n_beats=-1, **kwargs
    ):
        representations = super().get_representations(representation_types, return_arr=return_arr, **kwargs)
        if return_arr and n_beats > 0:
            for rep_type in ["beats_waveforms", "beats_features"]:
                if rep_type in representation_types:
                    actual_n_beats = representations[rep_type].shape[0]
                    if n_beats < actual_n_beats:
                        representations[rep_type] = representations[rep_type][:n_beats]
                    elif n_beats > actual_n_beats:
                        n_more_beats = n_beats - actual_n_beats
                        pad_width = ((0, n_more_beats), (0, 0), (0, 0))
                        representations[rep_type] = np.pad(representations[rep_type], pad_width)
        return representations
