"""Module used to provide data for PTB-XL dataset"""

import ast
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.preprocessing import LabelEncoder

from data.representations import (
    create_dataset,
    get_periodic_representations,
    load_split,
)
from signals.ecg import create_multichannel_ecg
from utils import DATA_PATH

PTBXL_PATH = DATA_PATH / "ptbxl"
RAW_DATASET_PATH = PTBXL_PATH / "raw"
RAW_TENSORS_PATH = PTBXL_PATH / "raw_tensors"
RAW_TENSORS_DATA_PATH = RAW_TENSORS_PATH / "data"
TARGETS_PATH = RAW_TENSORS_PATH / "targets"

PTBXL_ECG_FS = 100
PTBXL_TARGET = "diagnostic_class"


def load_raw_ptbxl_data(fs: float = PTBXL_ECG_FS, target: str = PTBXL_TARGET):
    """Load raw PTB-XL data.

    Args:
        fs (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.
    """

    def load_raw_data(df, fs, path):
        if fs == 100:
            data = [wfdb.rdsamp(str(path / f)) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(str(path / f)) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data

    def aggregate_class(y_dic, class_name):
        tmp = []
        for key, val in y_dic.items():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key][class_name])
            tmp = list(set(tmp))
            if len(tmp) == 1:
                diagnostic_class = tmp[0]
            else:
                diagnostic_class = np.nan
        return diagnostic_class

    # load and convert annotation data
    Y = pd.read_csv(RAW_DATASET_PATH / "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data(Y, fs, RAW_DATASET_PATH)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(RAW_DATASET_PATH / "scp_statements.csv", index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    Y["diagnostic_class"] = Y.scp_codes.apply(lambda codes: aggregate_class(codes, "diagnostic_class"))
    Y["diagnostic_subclass"] = Y.scp_codes.apply(lambda codes: aggregate_class(codes, "diagnostic_subclass"))

    # Split data into train and test
    val_fold = 9
    test_fold = 10

    target_mask = ~pd.isnull(Y[target]).values

    val_mask = (Y["strat_fold"] == val_fold).values & target_mask
    test_mask = (Y["strat_fold"] == test_fold).values & target_mask
    train_mask = ~(val_mask | test_mask) & target_mask

    dataset = {
        "train": {"X": X[train_mask], "y": Y[train_mask][target].values},
        "val": {"X": X[val_mask], "y": Y[val_mask][target].values},
        "test": {"X": X[test_mask], "y": Y[test_mask][target].values},
    }

    return dataset


def load_raw_ptbxl_data_and_save_to_tensors(fs: float = PTBXL_ECG_FS, target: str = PTBXL_TARGET, encode_labels=True):
    """Load raw PTB-XL data (as downloaded from physionet) and save it to tensors.

    Waveforms are saved as torch `.pt` files.
    Labels and classes mappings are saved as numpy  `.npy` files.

    Args:
        fs (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.
        encode_labels (bool): Whether to use `LabelEncoder` to encode labels. Defaults to `True`.
    """
    TARGETS_PATH = TARGETS_PATH / target
    TARGETS_PATH.mkdir(parents=True, exist_ok=True)

    RAW_TENSORS_DATA_PATH.mkdir(parents=True, exist_ok=True)

    ptbxl_data = load_raw_ptbxl_data(fs, target)
    if encode_labels:
        label_encoder = LabelEncoder().fit(ptbxl_data["train"]["y"])
        ptbxl_data["train"]["y"] = label_encoder.transform(ptbxl_data["train"]["y"])
        ptbxl_data["val"]["y"] = label_encoder.transform(ptbxl_data["val"]["y"])
        ptbxl_data["test"]["y"] = label_encoder.transform(ptbxl_data["test"]["y"])
        np.save(TARGETS_PATH / "classes.npy", label_encoder.classes_)

    for split in ["train", "val", "test"]:
        data_tensor = torch.tensor(ptbxl_data[split]["X"])
        targets_npy = ptbxl_data[split]["y"]
        torch.save(data_tensor, DATA_PATH / f"{split}_data.pt")
        np.save(TARGETS_PATH / f"{split}_targets.npy", targets_npy)


def get_ptbxl_representation(
    channels_data: torch.Tensor,
    fs: float = PTBXL_ECG_FS,
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


def create_ptbxl_dataset(representation_types: List[str], fs: float = PTBXL_ECG_FS):
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
    create_dataset(
        raw_tensors_path=RAW_TENSORS_DATA_PATH,
        representations_path=PTBXL_PATH / f"representations_{fs}",
        get_repr_func=get_ptbxl_representation,
        **params,
    )


def load_ptbxl_split(
    split: str, representation_type: str, fs: float = PTBXL_ECG_FS, target: str = PTBXL_TARGET
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
        representations_path=PTBXL_PATH / f"representations_{fs}",
        targets_path=TARGETS_PATH / target,
        representation_type=representation_type,
    )
