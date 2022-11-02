import glob
import logging
from collections import OrderedDict
from typing import List, Tuple

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from msr.data.namespace import DATA_PATH

SLEEP_EDF_PATH = DATA_PATH / "sleepEDF"
RAW_DATASET_PATH = SLEEP_EDF_PATH / "raw"
RAW_CSV_PATH = SLEEP_EDF_PATH / "raw_csv"
LOGS_PATH = SLEEP_EDF_PATH / "logs"

RAW_INFO_PATH = LOGS_PATH / "raw_info.csv"
SPLIT_INFO_PATH = LOGS_PATH / "split_info.csv"

RAW_TENSORS_PATH = SLEEP_EDF_PATH / "raw_tensors"
RAW_TENSORS_DATA_PATH = RAW_TENSORS_PATH / "data"
RAW_TENSORS_TARGETS_PATH = RAW_TENSORS_PATH / "targets"
REPRESENTATIONS_PATH = SLEEP_EDF_PATH / f"representations"

EVENT_ID = OrderedDict({f"Sleep stage {tok}": i for i, tok in enumerate(["W", 1, 2, 3, 4, "R", "M"])})
INV_EVENT_ID = OrderedDict({v: k for k, v in EVENT_ID.items()})


def get_sleep_edf_raw_data_info() -> pd.DataFrame:
    """TODO

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
    info.to_csv(RAW_INFO_PATH, index=False)
    return info


def parse_edf_to_epochs_dataframe(
    data_file: str, header_file: str, sample_len_sec: float = 30, verbose: bool = False
) -> pd.DataFrame:
    """From MNE tutorial: https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html
    # TODO
    Args:
        data_file (str): _description_
        header_file (str): _description_
        sample_len_sec (float, optional): _description_. Defaults to 30.
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
    df.loc[:, ["target"]] = np.array([EVENT_ID[target] for target in df["condition"].values])  # encode targets
    # only EEG data is needed
    return df[["target", "EEG Fpz-Cz", "EEG Pz-Oz"]]


def create_sleep_edf_raw_csv_dataset(sample_len_sec: float = 30, verbose: bool = False):
    """TODO

    Args:
        sample_len_sec (float, optional): _description_. Defaults to 30.
        verbose (bool, optional): _description_. Defaults to False.
    """
    info = pd.read_csv(RAW_INFO_PATH)
    for row in tqdm(info.iterrows(), total=len(info)):
        row = row[1]
        data_file, header_file, subject, night = row[["psg", "hypnogram", "subject", "night"]]
        epochs_df = parse_edf_to_epochs_dataframe(
            data_file, header_file, sample_len_sec=sample_len_sec, verbose=verbose
        )
        epochs_df.to_csv(RAW_CSV_PATH / f"{subject}_{night}.csv")


# TODO: use version from utils
def create_train_val_test_split_info(random_state: int = 42):
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


def create_raw_tensors_dataset():
    """TODO

    Args:
        splits (List[str], optional): _description_. Defaults to ["train"].
    """
    splits = ["train", "val", "test"]
    for split in tqdm(splits, desc="Splits"):
        split_info = pd.read_csv(SPLIT_INFO_PATH).query(f"split == '{split}'")
        data_rows = []
        for row in tqdm(split_info.iterrows(), total=len(split_info), desc=split):
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
        all_splits = np.concatenate([[split] * n_epochs[i] for i in range(len(n_epochs))])

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

        torch.save(all_data, RAW_TENSORS_DATA_PATH / f"{split}.pt")
        np.save(RAW_TENSORS_TARGETS_PATH / f"{split}.npy", all_targets)

        info = np.array(list(EVENT_ID.keys()))
        np.save(RAW_TENSORS_TARGETS_PATH / "info.npy", info)

        epochs_info.to_csv(LOGS_PATH / f"{split}_epochs_info.csv", index=False)


@hydra.main(version_base=None, config_path="../../configs/data", config_name="raw")
def main(cfg: DictConfig):
    cfg = cfg.sleep_edf
    log.info(cfg)

    paths = [
        SLEEP_EDF_PATH,
        RAW_DATASET_PATH,
        RAW_CSV_PATH,
        RAW_TENSORS_PATH,
        RAW_TENSORS_DATA_PATH,
        RAW_TENSORS_TARGETS_PATH,
        REPRESENTATIONS_PATH,
        LOGS_PATH,
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    get_sleep_edf_raw_data_info()

    create_sleep_edf_raw_csv_dataset(sample_len_sec=cfg.sample_len_sec, verbose=cfg.verbose)

    create_train_val_test_split_info(random_state=cfg.random_state)

    create_raw_tensors_dataset()


if __name__ == "__main__":
    main()

# TODO: add files download from web, then remove it on the fly after csvs creation
# TODO: check if works
# TODO: get rid of raw_csv directory
# TODO: Add docstrings
