from typing import List, Type, Union

import numpy as np

from msr.signals.base import MultiChannelPeriodicSignal, MultiChannelSignal


class RepresentationExtractor:
    def __init__(self, measurement: Type[MultiChannelSignal]):
        self.measurement = measurement

    def set_windows(self, win_len_s, step_s):
        self.measurement.set_windows(win_len_s, step_s)

    def get_representations(self, representation_types: List[str], return_arr: bool = True, **kwargs):
        if representation_types == "all":
            representation_types = list(self.measurement.representations_funcs.keys())
        return {
            rep_type: self.measurement.representations_funcs[rep_type](return_arr=return_arr, **kwargs)
            for rep_type in representation_types
        }

    def get_feature_names(self, representation_types: List[str]):
        if representation_types == "all":
            representation_types = list(self.measurement.representations_funcs.keys())
        feature_names = {}
        for rep_type in representation_types:
            if rep_type in self.measurement.feature_names_funcs:
                feature_names[rep_type] = self.measurement.feature_names_funcs[rep_type]()
        return feature_names


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
                        if rep_type == "beats_waveforms":  # beats waveforms has shape [B, N, C]
                            pad_width = ((0, n_more_beats), (0, 0), (0, 0))
                        else:  # beats features has shape [B, F]
                            pad_width = ((0, n_more_beats), (0, 0))
                        representations[rep_type] = np.pad(representations[rep_type], pad_width)
        return representations


def create_representation_extractor(
    multichannel_signal: Type[Union[MultiChannelPeriodicSignal, MultiChannelSignal]],
    windows_params: dict,
    beats_params: dict = {},
    agg_beat_params: dict = {},
):
    is_periodic = isinstance(multichannel_signal, MultiChannelPeriodicSignal)
    RepExtractorClass = PeriodicRepresentationExtractor if is_periodic else RepresentationExtractor
    rep_extractor = RepExtractorClass(multichannel_signal)
    if windows_params:
        rep_extractor.set_windows(**windows_params)
    if beats_params and is_periodic:
        rep_extractor.set_beats(**beats_params)
        if agg_beat_params:
            rep_extractor.set_agg_beat(**agg_beat_params)
    return rep_extractor
