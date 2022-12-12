from typing import List, Literal


class BaseNamespace:
    @classmethod
    def all(cls) -> List[str]:
        restricted_chars = ["__", "all", "None", ".", "dtype"]
        attrs = [
            attr for name, attr in cls.__dict__.items() if all([chars not in str(name) for chars in restricted_chars])
        ]
        return attrs

    @classmethod
    def get_type(cls):
        return List[Literal[tuple(cls.all())]]


class Representation(BaseNamespace):
    WHOLE_SIGNAL_WAVEFORMS = "whole_signal_waveforms"
    WHOLE_SIGNAL_FEATURS = "whole_signal_features"
    WINDOWS_WAVEFORMS = "windows_waveforms"
    WINDOWS_FEATURES = "windows_features"


class PeriodicRepresentation(BaseNamespace):
    WHOLE_SIGNAL_WAVEFORMS = "whole_signal_waveforms"
    WHOLE_SIGNAL_FEATURS = "whole_signal_features"
    WINDOWS_WAVEFORMS = "windows_waveforms"
    WINDOWS_FEATURES = "windows_features"
    BEATS_WAVEFORMS = "beats_waveforms"
    BEATS_FEATURES = "beats_features"
    AGG_BEAT_WAVEFORMS = "agg_beat_waveforms"
    AGG_BEAT_FEATURES = "agg_beat_features"


def get_setters_mask(representation_types: List[str]):
    set_windows = (
        PeriodicRepresentation.WINDOWS_WAVEFORMS in representation_types
        or PeriodicRepresentation.WINDOWS_FEATURES in representation_types
    )
    set_agg_beat = (
        PeriodicRepresentation.AGG_BEAT_WAVEFORMS in representation_types
        or PeriodicRepresentation.AGG_BEAT_FEATURES in representation_types
    )
    set_beats = (
        PeriodicRepresentation.BEATS_WAVEFORMS in representation_types
        or PeriodicRepresentation.BEATS_FEATURES in representation_types
        or set_agg_beat
    )
    return set_beats, set_windows, set_agg_beat
