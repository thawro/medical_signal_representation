import glob
import logging
from collections import OrderedDict
from typing import List, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from msr.utils import DATA_PATH, load_tensor, run_in_parallel_with_joblib

DATASET_NAME = "sleep_edf"
ZIP_FILE_URL = "https://www.physionet.org/static/published-projects/sleep-edfx/sleep-edf-database-expanded-1.0.0.zip"
DATASET_PATH = DATA_PATH / DATASET_NAME
RAW_DATASET_PATH = DATASET_PATH / "raw"
SLEEP_CASETTE_PATH = RAW_DATASET_PATH / "sleep-cassette"
RAW_CSV_PATH = RAW_DATASET_PATH / "csv"
LOGS_PATH = DATASET_PATH / "logs"

RAW_INFO_PATH = LOGS_PATH / "raw_info.csv"
SPLIT_INFO_PATH = LOGS_PATH / "split_info.csv"

RAW_TENSORS_PATH = DATASET_PATH / "raw_tensors"
RAW_TENSORS_DATA_PATH = RAW_TENSORS_PATH / "data"
RAW_TENSORS_TARGETS_PATH = RAW_TENSORS_PATH / "targets"
for path in [RAW_TENSORS_DATA_PATH, RAW_TENSORS_TARGETS_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


EVENT_ID = OrderedDict({f"Sleep stage {tok}": i for i, tok in enumerate(["W", 1, 2, 3, 4, "R", "M"])})
INV_EVENT_ID = OrderedDict({v: k for k, v in EVENT_ID.items()})
SIG_NAMES = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]


def get_sleep_edf_raw_data_info() -> pd.DataFrame:
    """Create csv with summarized info about data samples.

    Info includes subject, night, hypnogram_file, psg_file, raw_csv_file

    Returns:
        pd.DataFrame: DataFrame with info.
    """
    log.info(f"Creating info csv ({str(RAW_INFO_PATH)}).")
    all_files = glob.glob(f"{SLEEP_CASETTE_PATH}/*")
    all_files = [file.split("/")[-1] for file in all_files]

    # sort to make sure the subject nights are in correct order
    hypnogram_files = sorted([file for file in all_files if "Hypnogram" in file])
    psg_files = sorted([file for file in all_files if "PSG" in file])

    info = []
    for hypnogram, psg in zip(hypnogram_files, psg_files):
        subject, night = int(hypnogram[3:5]), int(hypnogram[5])
        info.append(dict(subject=subject, night=night, hypnogram=hypnogram, psg=psg))
    info = pd.DataFrame(info)
    info["sample_id"] = [f"{subject}_{night}" for subject, night in info[["subject", "night"]].values]
    info.to_csv(RAW_INFO_PATH, index=False)
    return info


def parse_edf_file_to_df(
    data_file: str, header_file: str, sample_len_sec: float, sig_names: List[str], verbose: bool = False
) -> pd.DataFrame:
    """From MNE tutorial: https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html

    Parse edf hypnogram (header) and psg (data) files into dataframe with signals and targets.
    Args:
        data_file (str): PSG filepath
        header_file (str): Hypnogram filepath
        sample_len_sec (float): Length of single sample in seconds (used to create epochs). Defaults to 30.
        verbose (bool): Whether to show additional info from mne package. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with signals, epochs and targets.
    """
    raw_data = mne.io.read_raw_edf(data_file, stim_channel="Event marker", misc=["Temp rectal"], verbose=verbose)
    fs = raw_data.info["sfreq"]
    targets = mne.read_annotations(header_file)

    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    left_mins_margin, right_mins_margin = 30, 30
    targets.crop(
        targets[1]["onset"] - left_mins_margin * 60, targets[-2]["onset"] + right_mins_margin * 60, verbose=verbose
    )
    raw_data.set_annotations(targets, emit_warning=False, verbose=verbose)

    events, _ = mne.events_from_annotations(raw_data, event_id=EVENT_ID, chunk_duration=sample_len_sec, verbose=verbose)

    tmax = sample_len_sec - 1.0 / fs  # tmax in included

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
    df["target"] = np.array([EVENT_ID[target] for target in df["condition"].values])  # encode targets
    return df[["target", *sig_names]]


def create_raw_csv_dataset(sample_len_sec: float, sig_names: List[str], verbose: bool = False):
    """Parse EDF files to DataFrames and save it as csv files.

    There is a single csv file for each subject and each night.

    Args:
        sample_len_sec (float): Length of single sample in seconds (used to create epochs). Defaults to 30.
        verbose (bool, optional): Whether to show additional info from mne package. Defaults to False.
    """
    RAW_CSV_PATH.mkdir(parents=True, exist_ok=True)
    log.info(f"Creating raw csv dataset (located in {RAW_CSV_PATH})")
    info = pd.read_csv(RAW_INFO_PATH)

    def save_edf_to_df(subject, night, hypnogram, psg):
        epochs_df = parse_edf_file_to_df(
            data_file=SLEEP_CASETTE_PATH / psg,
            header_file=SLEEP_CASETTE_PATH / hypnogram,
            sig_names=sig_names,
            sample_len_sec=sample_len_sec,
            verbose=verbose,
        )
        epochs_df.to_csv(RAW_CSV_PATH / f"{subject}_{night}.csv")

    run_in_parallel_with_joblib(
        fn=save_edf_to_df, iterable=info.values[:, :4], n_jobs=8, desc="Saving edf files to csv"
    )


def load_raw_csv_to_df(subject: int, night: int) -> pd.DataFrame:
    """Load csv to DataFrame for specific subject and night.

    Args:
        subject (int): Subject code.
        night (int): Night code.

    Returns:
        pd.DataFrame: DataFrame with signals, epochs and targets.
    """
    return pd.read_csv(RAW_CSV_PATH / f"{subject}_{night}.csv").set_index("epoch")


def cut_df_into_data_and_targets_epochs(data: pd.DataFrame) -> Tuple[torch.Tensor, np.ndarray]:
    """Cut DataFrame (from whole night) into epochs and return corresponding signals data and targets

    Args:
        data (pd.DataFrame): DataFrame with signals, epochs and targets.

    Returns:
        Tuple[torch.Tensor, np.ndarray]: (data_tensor, targets)
    """
    # remove last epoch if it has less samples
    data_df = data.drop("target", axis=1)
    epochs = data.index.unique().values
    first_epoch, last_epoch = epochs[0], epochs[-1]

    last_epoch_samples = len(data.loc[last_epoch])
    first_epoch_samples = len(data.loc[first_epoch])
    if last_epoch_samples < first_epoch_samples:
        data = data.loc[:last_epoch]

    data_tensor = torch.from_numpy(np.array([data_df.loc[epoch].values.T for epoch in epochs]))
    targets = torch.tensor([data.loc[epoch, "target"].values[0] for epoch in data.index.unique()])

    return data_tensor, targets


def create_raw_tensors_dataset():
    """Create raw tensors dataset

    Each subject_night DataFrame is cut into epochs and epochs are concatenated for each split to create dataset.
    """
    log.info("Creating raw tensors dataset")
    splits = ["train", "val", "test"]
    for split in tqdm(splits, desc="Splits"):
        split_info = pd.read_csv(SPLIT_INFO_PATH).query(f"split == '{split}'")

        def get_data_dict(subject, night):
            night_data_df = load_raw_csv_to_df(subject, night)
            data, targets = cut_df_into_data_and_targets_epochs(night_data_df)
            epochs = torch.arange(len(targets))
            return dict(data=data, targets=targets, epochs=epochs, subject=subject, night=night)

        data_rows = run_in_parallel_with_joblib(
            fn=get_data_dict, iterable=split_info.values[:, :2], n_jobs=8, desc=split
        )

        n_epochs = torch.tensor([len(data_row["targets"]) for data_row in data_rows])
        all_data = torch.cat([data_row["data"] for data_row in data_rows])
        all_targets = torch.cat([data_row["targets"] for data_row in data_rows])
        all_targets_names = np.array([INV_EVENT_ID[target.item()] for target in all_targets])
        all_epochs = torch.cat([data_row["epochs"] for data_row in data_rows])
        all_subjects = torch.cat(
            [torch.ones(n_epochs[i]).int() * data_rows[i]["subject"] for i in range(len(n_epochs))]
        )
        all_nights = torch.cat([torch.ones(n_epochs[i]).int() * data_rows[i]["night"] for i in range(len(n_epochs))])
        all_splits = np.concatenate([[split] * n_epochs[i] for i in range(len(n_epochs))])

        epochs_info = pd.DataFrame(
            {
                "subject": all_subjects,
                "night": all_nights,
                "split": all_splits,
                "target": all_targets,
                "target_name": all_targets_names,
                "epoch": all_epochs,
            }
        )

        torch.save(all_data, RAW_TENSORS_DATA_PATH / f"{split}.pt")
        torch.save(all_targets, RAW_TENSORS_TARGETS_PATH / f"{split}.pt")

        info = np.array(list(EVENT_ID.keys()))
        np.save(RAW_TENSORS_TARGETS_PATH / "info.npy", info)
        epochs_info.to_csv(LOGS_PATH / f"{split}_epochs_info.csv", index=False)


def load_sleep_edf_raw_data(split):
    return load_tensor(RAW_TENSORS_DATA_PATH / f"{split}.pt")
