import glob
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

from data.representations import (
    create_dataset,
    get_periodic_representations,
    get_representations,
    load_split,
)
from signals.eeg import create_multichannel_eeg
from utils import DATA_PATH

SLEEP_EDF_PATH = DATA_PATH / "sleepEDF"
RAW_DATASET_PATH = SLEEP_EDF_PATH / "raw"
RAW_CSV_PATH = SLEEP_EDF_PATH / "raw_csv"

RAW_INFO_PATH = SLEEP_EDF_PATH / "raw_info.csv"
SPLIT_INFO_PATH = SLEEP_EDF_PATH / "split_info.csv"

RAW_TENSORS_PATH = SLEEP_EDF_PATH / "raw_tensors"
RAW_TENSORS_DATA_PATH = RAW_TENSORS_PATH / "data"
RAW_TENSORS_TARGETS_PATH = RAW_TENSORS_PATH / "targets"
REPRESENTATIONS_PATH = SLEEP_EDF_PATH / f"representations"
SLEEP_EDF_EEG_FS = 100

EVENT_ID = OrderedDict(
    {
        "Sleep stage W": 0,
        "Sleep stage 1": 1,
        "Sleep stage 2": 2,
        "Sleep stage 3": 3,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
        "Sleep stage M": 6,
    }
)
INV_EVENT_ID = OrderedDict({v: k for k, v in EVENT_ID.items()})


def get_sleep_edf_raw_data_info(save: bool = False) -> pd.DataFrame:
    """TODO

    Args:
        save (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    all_files = glob.glob(f"{RAW_DATASET_PATH}/*")
    all_files = [file.split("/")[-1] for file in all_files]

    # sort to make sure the subject nights are in correct order
    hypnogram_files = sorted([file for file in all_files if "Hypnogram" in file])
    psg_files = sorted([file for file in all_files if "PSG" in file])

    info = []
    for hypnogram, psg in tqdm(zip(hypnogram_files, psg_files), desc="Creating info"):
        subject, night = hypnogram[3:5], hypnogram[5]
        info.append(
            {
                "subject": subject,
                "night": night,
                "hypnogram": hypnogram,
                "psg": psg,
                "raw_csv_path": f"{subject}/{night}.csv",
            }
        )
    info = pd.DataFrame(info)
    if save:
        info.to_csv(RAW_INFO_PATH, index=False)
    return info


def parse_edf_to_epochs_dataframe(
    data_file: str, header_file: str, epoch_duration: float = 30, verbose: bool = False
) -> pd.DataFrame:
    """From MNE tutorial: https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html
    # TODO
    Args:
        data_file (str): _description_
        header_file (str): _description_
        epoch_duration (float, optional): _description_. Defaults to 30.
        verbose (bool, optional): _description_. Defaults to False.

    Returns:
        pd.DataFrame: _description_
    """
    raw_data = mne.io.read_raw_edf(
        RAW_DATASET_PATH / data_file, stim_channel="Event marker", misc=["Temp rectal"], verbose=verbose
    )
    fs = raw_data.info["sfreq"]
    targets = mne.read_annotations(RAW_DATASET_PATH / header_file)

    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    targets.crop(targets[1]["onset"] - 30 * 60, targets[-2]["onset"] + 30 * 60, verbose=verbose)
    raw_data.set_annotations(targets, emit_warning=False, verbose=verbose)

    events, _ = mne.events_from_annotations(raw_data, event_id=EVENT_ID, chunk_duration=epoch_duration, verbose=verbose)

    tmax = epoch_duration - 1.0 / fs  # tmax in included

    epochs = mne.Epochs(
        raw=raw_data,
        events=events,
        event_id=EVENT_ID,
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        on_missing="warn",
        verbose=verbose,
    )
    df = epochs.to_data_frame(time_format=None, verbose=verbose).set_index("epoch")
    df.loc[:, ["target"]] = np.array([EVENT_ID[target] for target in df["condition"].values])  # encode targets
    # only EEG data is needed
    return df[["target", "EEG Fpz-Cz", "EEG Pz-Oz"]]


def create_sleep_edf_raw_csv_dataset(epoch_duration: float = 30, verbose: bool = False):
    """TODO

    Args:
        epoch_duration (float, optional): _description_. Defaults to 30.
        verbose (bool, optional): _description_. Defaults to False.
    """
    info = pd.read_csv(RAW_INFO_PATH)
    for row in tqdm(info.iterrows(), total=len(info)):
        row = row[1]
        data_file, header_file, subject, night = row[["psg", "hypnogram", "subject", "night"]]
        epochs_df = parse_edf_to_epochs_dataframe(
            data_file, header_file, epoch_duration=epoch_duration, verbose=verbose
        )
        epochs_df.to_csv(RAW_CSV_PATH / f"{subject}_{night}.csv")


def create_train_val_test_split_info(random_state: int = 42, save: bool = False):
    """TODO

    Args:
        random_state (int, optional): _description_. Defaults to 42.
        save (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    info = pd.read_csv(RAW_INFO_PATH)

    groups = info["subject"].values
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train_idx, val_test_idx = next(gss.split(info, groups=groups))

    train_info = info.iloc[train_idx]
    val_test_info = info.iloc[val_test_idx]

    val_test_groups = val_test_info["subject"].values
    gss = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(gss.split(val_test_info, groups=val_test_groups))

    val_info = val_test_info.iloc[val_idx]
    test_info = val_test_info.iloc[test_idx]

    train_info.loc[:, "split"] = "train"
    val_info.loc[:, "split"] = "val"
    test_info.loc[:, "split"] = "test"

    train_val_test_split_info = pd.concat([train_info, val_info, test_info])

    if save:
        train_val_test_split_info.to_csv(SPLIT_INFO_PATH, index=False)
    return train_val_test_split_info


def load_raw_csv_to_df(subject: int, night: int) -> pd.DataFrame:
    """TODO

    Args:
        subject (int): _description_
        night (int): _description_

    Returns:
        pd.DataFrame: _description_
    """
    return pd.read_csv(RAW_CSV_PATH / f"{str(subject).zfill(2)}_{night}.csv").set_index("epoch")


def get_data_and_targets(data: pd.DataFrame) -> Tuple[torch.Tensor, np.ndarray]:
    # remove last epoch if it has less samples
    data_df = data.drop("condition", axis=1)
    epochs = data.index.unique().values
    first_epoch, last_epoch = epochs[0], epochs[-1]
    n_epochs = last_epoch - first_epoch + 1
    n_signals = len(data_df.columns)

    last_epoch_samples = len(data.loc[last_epoch])
    first_epoch_samples = len(data.loc[first_epoch])
    if last_epoch_samples < first_epoch_samples:
        data = data.loc[:last_epoch]

    data_tensor = torch.from_numpy(np.array([data_df.loc[epoch].values.T for epoch in epochs]))
    targets = np.array([data.loc[epoch].condition.values[0] for epoch in data.index.unique()])

    return data_tensor, targets


def create_raw_tensors_dataset(split_types: List[str] = ["train"]):
    """TODO

    Args:
        split_types (List[str], optional): _description_. Defaults to ["train"].
    """
    for split_type in tqdm(split_types, desc="Splits"):
        split_info = pd.read_csv(SPLIT_INFO_PATH).query(f"split == '{split_type}'")
        data_rows = []
        for row in tqdm(split_info.iterrows(), total=len(split_info), desc=split_type):
            row = row[1]
            subject, night = row["subject"], row["night"]
            df = load_raw_csv_to_df(subject, night)
            data, targets = get_data_and_targets(df)
            data_rows.append(
                {
                    "data": data,
                    "targets": targets,
                    "epochs": np.arange(len(targets)),
                    "subject": subject,
                    "night": night,
                }
            )

        n_epochs = np.array([data_row["targets"].__len__() for data_row in data_rows])

        all_data = torch.cat([data_row["data"] for data_row in data_rows])
        all_targets = np.concatenate([data_row["targets"] for data_row in data_rows])
        all_targets_names = np.array([INV_EVENT_ID[label] for label in all_targets])
        all_epochs = np.concatenate([data_row["epochs"] for data_row in data_rows])
        all_subjects = np.concatenate(
            [np.ones(n_epochs[i]).astype(int) * data_rows[i]["subject"] for i in range(len(n_epochs))]
        )
        all_nights = np.concatenate(
            [np.ones(n_epochs[i]).astype(int) * data_rows[i]["night"] for i in range(len(n_epochs))]
        )
        all_splits = np.concatenate([[split_type] * n_epochs[i] for i in range(len(n_epochs))])

        epochs_info = pd.DataFrame(
            {
                "subject": all_subjects,
                "night": all_nights,
                "split": all_splits,
                "label": all_targets,
                "label_name": all_targets_names,
                "epoch": all_epochs,
            }
        )

        torch.save(all_data, RAW_TENSORS_DATA_PATH / f"{split_type}_data.pt")
        np.save(RAW_TENSORS_TARGETS_PATH / f"{split_type}_targets.npy", all_targets)

        info = np.array(list(EVENT_ID.keys()))
        np.save(RAW_TENSORS_TARGETS_PATH / "info.npy", info)

        epochs_info.to_csv(SLEEP_EDF_PATH / f"{split_type}_epochs_info.csv", index=False)


def get_sleep_edf_representation(
    channels_data: torch.Tensor,
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
    eeg = create_multichannel_eeg(channels_data.numpy(), fs=SLEEP_EDF_EEG_FS)
    return get_representations(eeg, windows_params, representation_types=representation_types)


def create_sleep_edf_dataset(representation_types):
    """Create and save data files (`.pt`) for all representations.

    Args:
        splits (list): Split types to create representations data for. Defaults to ['train', 'val', 'test'].
        fs (float): Sampling frequency of signals. Defaults to 100.
    """
    params = dict(
        representation_types=representation_types,
        beats_params=dict(source_channel=2),
        agg_beat_params=dict(valid_only=False),
        windows_params=dict(win_len_s=3, step_s=2),
    )
    create_dataset(
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
