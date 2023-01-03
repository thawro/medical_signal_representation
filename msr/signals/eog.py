import neurokit2 as nk

from msr.signals.base import MultiChannelSignal, Signal
from msr.signals.utils import parse_feats_to_array, parse_nested_feats


class EOGSignal(Signal):
    def __init__(self, name, data, fs, start_sec=0):
        super().__init__(name, data, fs, start_sec)
        self.units = "Voltage [V]"
        self.feature_extraction_funcs.update(
            {
                # TODO
            }
        )

    @property
    def cleaned(self):
        return nk.eog_clean(self.data, sampling_rate=self.fs)


class MultiChannelEOGSignal(MultiChannelSignal):
    def __init__(self, signals):
        super().__init__(signals)
