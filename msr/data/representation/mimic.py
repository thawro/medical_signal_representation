from functools import partial
from typing import Dict, List, Union

import numpy as np
import torch

from msr.data.measurements.mimic import MimicMeasurement
from msr.data.raw.mimic import DATASET_PATH, RAW_TENSORS_DATA_PATH, TARGETS_PATH
from msr.data.representation.utils import (
    create_representations_dataset,
    get_periodic_representations,
    load_split,
)

REPRESENTATIONS_PATH = DATASET_PATH / f"representations"


def get_mimic_representation(
    data: torch.Tensor,
    representation_types: List[str],
    beats_params: Dict[str, Union[str, float, int]],
    n_beats: int,
    agg_beat_params: Dict[str, Union[str, float, int]],
    windows_params: Dict[str, Union[str, float, int]],
    fs: float = 125,
    return_feature_names: bool = False,
):
    """Get all types of representations (returned by ECGSignal objects).
    Args:
        channels_data (torch.Tensor): ECG channels data of shape `[TODO]`, where TODO.
        fs (float): Sampling rate. Defaults to 100.
    Returns:
        Dict[str, torch.Tensor]: Dict with representations names as keys and `torch.Tensor` objects as values.
    """
    ppg, ecg = data.numpy()
    multichannel_mimic = MimicMeasurement(ppg, ecg, fs=fs)
    return get_periodic_representations(
        multichannel_mimic,
        representation_types=representation_types,
        beats_params=beats_params,
        n_beats=n_beats,
        agg_beat_params=agg_beat_params,
        windows_params=windows_params,
        return_feature_names=return_feature_names,
    )


def create_mimic_representations_dataset(
    representation_types: List[str],
    beats_params: Dict[str, Union[str, float, int]],
    n_beats: int,
    agg_beat_params: Dict[str, Union[str, float, int]],
    windows_params: Dict[str, Union[str, float, int]],
    fs: float = 100,
):
    """Create and save data files (`.pt`) for all representations.
    TODO
    Args:
        fs (float): Sampling frequency of signals. Defaults to 100.
    """
    get_repr_func = partial(
        get_mimic_representation,
        representation_types=representation_types,
        beats_params=beats_params,
        n_beats=n_beats,
        agg_beat_params=agg_beat_params,
        windows_params=windows_params,
        fs=fs,
    )
    create_representations_dataset(
        raw_tensors_path=RAW_TENSORS_DATA_PATH,
        representations_path=REPRESENTATIONS_PATH,
        get_repr_func=get_repr_func,
    )


def load_mimic_split(split: str, representation_type: str) -> Dict[str, np.ndarray]:
    return load_split(
        split=split,
        representations_path=REPRESENTATIONS_PATH,
        targets_path=TARGETS_PATH,
        representation_type=representation_type,
    )


# TODO: make it work
