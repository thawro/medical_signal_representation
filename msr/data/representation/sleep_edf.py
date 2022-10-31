from typing import Dict, List

import numpy as np
import torch
from mrs.data.raw.sleep_edf import (
    RAW_CSV_PATH,
    RAW_DATASET_PATH,
    RAW_INFO_PATH,
    RAW_TENSORS_DATA_PATH,
    RAW_TENSORS_PATH,
    RAW_TENSORS_TARGETS_PATH,
    REPRESENTATIONS_PATH,
    SLEEP_EDF_PATH,
    SPLIT_INFO_PATH,
)

from msr.data.representation.utils import (
    create_representations_dataset,
    get_representations,
    load_split,
)
from msr.signals.eeg import create_multichannel_eeg
from msr.utils import DATA_PATH


def get_sleep_edf_representation(
    channels_data: torch.Tensor,
    fs: float,
    representation_types: List[str] = ["whole_signal_waveforms"],
    windows_params=dict(win_len_s=3, step_s=2),
):
    """Get all types of representations (returned by ECGSignal objects).
    Args:
        channels_data (torch.Tensor): ECG channels data of shape `[TODO]`, where TODO.
        fs (float): Sampling rate. Defaults to 100.
    Returns:
        Dict[str, torch.Tensor]: Dict with representations names as keys and `torch.Tensor` objects as values.
    """
    eeg = create_multichannel_eeg(channels_data.numpy(), fs=fs)
    return get_representations(eeg, windows_params, representation_types=representation_types)


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
