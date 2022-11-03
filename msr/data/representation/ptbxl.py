"""Module used to provide data for PTB-XL dataset"""

from typing import Dict, List

import numpy as np
import torch

from msr.data.raw.ptbxl import DATASET_PATH, RAW_TENSORS_DATA_PATH, TARGETS_PATH
from msr.data.representation.utils import (
    create_representations_dataset,
    get_periodic_representations,
    load_split,
)
from msr.signals.ecg import create_multichannel_ecg


def get_ptbxl_representation(
    channels_data: torch.Tensor,
    fs: float,
    beats_params=dict(source_channel=2),
    agg_beat_params=dict(valid_only=False),
    windows_params=dict(win_len_s=3, step_s=2),
    representation_types: List[str] = ["whole_signal_waveforms"],
):
    """Get all types of representations (returned by ECGSignal objects).
    Args:
        channels_data (torch.Tensor): ECG channels data of shape `[TODO]`, where TODO.
        fs (float): Sampling rate. Defaults to 100.
    Returns:
        Dict[str, torch.Tensor]: Dict with representations names as keys and `torch.Tensor` objects as values.
    """
    ecg = create_multichannel_ecg(channels_data.T.numpy(), fs)
    return get_periodic_representations(
        ecg, beats_params, agg_beat_params, windows_params, representation_types=representation_types
    )


def create_ptbxl_representations_dataset(representation_types: List[str], fs: float = 100):
    """Create and save data files (`.pt`) for all representations.

    Args:
        splits (list): Split types to create representations data for. Defaults to ['train', 'val', 'test'].
        fs (float): Sampling frequency of signals. Defaults to 100.
    """
    params = dict(
        representation_types=representation_types,
        fs=fs,
        beats_params=dict(source_channel=2),
        agg_beat_params=dict(valid_only=False),
        windows_params=dict(win_len_s=3, step_s=2),
    )
    create_representations_dataset(
        raw_tensors_path=RAW_TENSORS_DATA_PATH,
        representations_path=DATASET_PATH / f"representations_{fs}",
        get_repr_func=get_ptbxl_representation,
        **params,
    )


def load_ptbxl_split(
    split: str, representation_type: str, fs: float = 100, target: str = "diagnostic_class"
) -> Dict[str, np.ndarray]:
    """Load PTB-XL train, val or test split for specified representation type, target and sampling rate (fs).

    What is loaded:
    * data (torch tensor)
    * targets (numpy array TODO: change it to tensor)
    * info (numpy array TODO: change it to tensor)
    Args:
        representation_type (str): Type of representation to load data for.
            One of `["whole_signal_waveforms", "whole_signal_features", "per_beat_waveforms",
            "per_beat_features", "whole_signal_embeddings"]
        target (str): Type of target for PTB-XL benchmark. One of `["diagnostic_class", "diagnostic_subclass"]`.
        split (str): Type of data split. One of `["train", "val", "test"]`.
        fs (float): Sampling rate of PTB-XL waveform data. One of `[100, 500]`. Defaults to 100.

    Returns:
        Dict[str, np.ndarray]: Dict with data, labels and classes mapper.
    """
    return load_split(
        split=split,
        representations_path=DATASET_PATH / f"representations_{fs}",
        targets_path=TARGETS_PATH / target,
        representation_type=representation_type,
    )
