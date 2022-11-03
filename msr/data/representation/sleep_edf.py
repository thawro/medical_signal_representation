from collections import OrderedDict
from typing import Dict, List

import numpy as np
import torch

from msr.data.raw.sleep_edf import (
    DATASET_PATH,
    RAW_TENSORS_DATA_PATH,
    RAW_TENSORS_TARGETS_PATH,
)
from msr.data.representation.utils import (
    create_representations_dataset,
    get_representations,
    load_split,
)
from msr.signals.base import MultiChannelSignal
from msr.signals.eeg import EEGSignal, create_multichannel_eeg
from msr.signals.eog import EOGSignal

REPRESENTATIONS_PATH = DATASET_PATH / f"representations"


def create_multichannel_sleep_edf(data, fs):
    signals = OrderedDict(
        {
            0: EEGSignal("EEG Fpz-Cz", data[0], fs),
            1: EEGSignal("EEG Pz-Oz", data[1], fs),
            2: EOGSignal("EOG horizontal", data[2], fs),
        }
    )
    return MultiChannelSignal(signals)


def get_sleep_edf_representation(
    data: torch.Tensor,
    fs: float,
    representation_types: List[str] = ["whole_signal_waveforms"],
    windows_params=dict(win_len_s=3, step_s=2),
):
    """Get all types of representations (returned by ECGSignal objects).
    Args:
        data (torch.Tensor): ECG channels data of shape `[TODO]`, where TODO.
        fs (float): Sampling rate. Defaults to 100.
    Returns:
        Dict[str, torch.Tensor]: Dict with representations names as keys and `torch.Tensor` objects as values.
    """
    multichannel_sleep_edf = create_multichannel_sleep_edf(data.numpy(), fs)
    return get_representations(multichannel_sleep_edf, windows_params, representation_types)


def create_sleep_edf_representations_dataset(representation_types: List[str], fs: float = 100):
    """Create and save data files (`.pt`) for all representations.

    Args:
        splits (list): Split types to create representations data for. Defaults to ['train', 'val', 'test'].
        fs (float): Sampling frequency of signals. Defaults to 100.
    """
    params = dict(
        fs=fs,
        representation_types=representation_types,
        windows_params=dict(win_len_s=3, step_s=2),
    )
    create_representations_dataset(
        raw_tensors_path=RAW_TENSORS_DATA_PATH,
        representations_path=REPRESENTATIONS_PATH,
        get_repr_func=get_sleep_edf_representation,
        **params,
    )


def load_sleep_edf_split(split: str, representation_type: str) -> Dict[str, np.ndarray]:
    return load_split(
        split=split,
        representations_path=REPRESENTATIONS_PATH,
        targets_path=RAW_TENSORS_TARGETS_PATH,
        representation_type=representation_type,
    )
