from collections import OrderedDict
from functools import partial
from typing import Dict, List, Union

import numpy as np
import torch

from msr.data.create_representations.utils import (
    create_representations_dataset,
    get_representations,
    load_split,
)
from msr.data.download.sleep_edf import (
    DATASET_PATH,
    RAW_TENSORS_DATA_PATH,
    RAW_TENSORS_TARGETS_PATH,
)
from msr.data.measurements.sleep_edf import SleepEDFMeasurement

REPRESENTATIONS_PATH = DATASET_PATH / f"representations"


def get_sleep_edf_representation(
    data: torch.Tensor,
    representation_types: List[str],
    windows_params: Dict[str, Union[str, float, int]],
    fs: float = 100,
    return_feature_names: bool = False,
):
    """Get all types of representations (returned by ECGSignal objects).
    Args:
        data (torch.Tensor): ECG channels data of shape `[TODO]`, where TODO.
        fs (float): Sampling rate. Defaults to 100.
    Returns:
        Dict[str, torch.Tensor]: Dict with representations names as keys and `torch.Tensor` objects as values.
    """
    eeg_0, eeg_1, eog = data.numpy()
    multichannel_sleep_edf = SleepEDFMeasurement(eeg_0, eeg_1, eog, fs)
    return get_representations(
        multichannel_sleep_edf,
        representation_types=representation_types,
        windows_params=windows_params,
        return_feature_names=return_feature_names,
    )


def create_sleep_edf_representations_dataset(
    representation_types: List[str], windows_params: Dict[str, Union[str, float, int]], fs: float = 100, n_jobs: int = 1
):
    """Create and save data files (`.pt`) for all representations.
    TODO
    Args:
        fs (float): Sampling frequency of signals. Defaults to 100.
    """
    get_repr_func = partial(
        get_sleep_edf_representation,
        representation_types=representation_types,
        windows_params=windows_params,
        fs=fs,
    )
    create_representations_dataset(
        raw_tensors_path=RAW_TENSORS_DATA_PATH,
        representations_path=REPRESENTATIONS_PATH,
        get_repr_func=get_repr_func,
        n_jobs=n_jobs,
    )


def load_sleep_edf_split(split: str, representation_type: str) -> Dict[str, np.ndarray]:
    return load_split(
        split=split,
        representations_path=REPRESENTATIONS_PATH,
        targets_path=RAW_TENSORS_TARGETS_PATH,
        representation_type=representation_type,
    )
