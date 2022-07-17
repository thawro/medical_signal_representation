import glob
from collections import OrderedDict
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm

SLEEP_EDF_PATH = Path("./../../data/sleepEDF")
RAW_DATASET_PATH = SLEEP_EDF_PATH / "raw"
RAW_CSV_PATH = SLEEP_EDF_PATH / "raw_csv"
RAW_TENSORS_PATH = SLEEP_EDF_PATH / "raw_tensors"

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


def create_raw_data_info(save=False):
    all_files = glob.glob(f"{RAW_DATASET_PATH}/*")
    all_files = [file.split("/")[-1] for file in all_files]

    # sort to make sure the subject nights are in correct order
    hypnogram_files = sorted([file for file in all_files if "Hypnogram" in file])
    psg_files = sorted([file for file in all_files if "PSG" in file])

    info = []
    for hypnogram, psg in zip(hypnogram_files, psg_files):
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
        info.to_csv(SLEEP_EDF_PATH / "raw_info.csv", index=False)

    return info


def parse_edf_to_epochs_dataframe(data_file, header_file, epoch_duration=30, verbose=False):
    data_file = RAW_DATASET_PATH / data_file
    header_file = RAW_DATASET_PATH / header_file

    raw_data = mne.io.read_raw_edf(data_file, stim_channel="Event marker", misc=["Temp rectal"], verbose=verbose)
    fs = raw_data.info["sfreq"]
    labels = mne.read_annotations(header_file)

    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    labels.crop(labels[1]["onset"] - 30 * 60, labels[-2]["onset"] + 30 * 60, verbose=verbose)
    raw_data.set_annotations(labels, emit_warning=False, verbose=verbose)

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

    # make csv smaller
    df.loc[:, ["condition"]] = np.array([EVENT_ID[condition] for condition in df["condition"].values])  # encode labels
    # drop time since we know fs and epoch length
    # only EEG data is needed
    return df[["condition", "EEG Fpz-Cz", "EEG Pz-Oz"]]


def create_sleep_edf_raw_csv_dataset(epoch_duration=30, verbose=False):
    info = pd.read_csv(SLEEP_EDF_PATH / "raw_info.csv")
    for row in tqdm(info.iterrows(), total=len(info)):
        row = row[1]
        data_file, header_file, subject, night = row[["psg", "hypnogram", "subject", "night"]]
        epochs_df = parse_edf_to_epochs_dataframe(
            data_file, header_file, epoch_duration=epoch_duration, verbose=verbose
        )
        epochs_df.to_csv(RAW_CSV_PATH / f"{subject}_{night}.csv")


def create_train_val_test_split_info(random_state=42, save=False):
    info = pd.read_csv(SLEEP_EDF_PATH / "raw_info.csv")

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
        train_val_test_split_info.to_csv(SLEEP_EDF_PATH / "split_info.csv", index=False)
    return train_val_test_split_info


def load_raw_csv_to_df(subject, night):
    return pd.read_csv(RAW_CSV_PATH / f"{str(subject).zfill(2)}_{night}.csv").set_index("epoch")


def get_data_and_labels_arrays(df):
    # remove last epoch if it has less samples
    data_df = df.drop("condition", axis=1)
    epochs = df.index.unique().values
    first_epoch, last_epoch = epochs[0], epochs[-1]
    n_epochs = last_epoch - first_epoch + 1
    n_signals = len(data_df.columns)

    last_epoch_samples = len(df.loc[last_epoch])
    first_epoch_samples = len(df.loc[first_epoch])
    if last_epoch_samples < first_epoch_samples:
        df = df.loc[:last_epoch]

    data_tensor = torch.from_numpy(np.array([data_df.loc[epoch].values.T for epoch in epochs]))
    labels = np.array([df.loc[epoch].condition.values[0] for epoch in df.index.unique()])

    return data_tensor, labels


def create_raw_tensors_dataset(split_types=["train"]):
    for split_type in tqdm(split_types, desc="Splits"):
        split_info = pd.read_csv(SLEEP_EDF_PATH / "split_info.csv").query(f"split == '{split_type}'")
        data_rows = []
        for row in tqdm(split_info.iterrows(), total=len(split_info), desc=split_type):
            row = row[1]
            subject, night = row["subject"], row["night"]
            df = load_raw_csv_to_df(subject, night)
            data, labels = get_data_and_labels_arrays(df)
            data_rows.append(
                {
                    "data": data,
                    "labels": labels,
                    "epochs": np.arange(len(labels)),
                    "subject": subject,
                    "night": night,
                }
            )

        n_epochs = np.array([data_row["labels"].__len__() for data_row in data_rows])

        all_data = torch.cat([data_row["data"] for data_row in data_rows])
        all_labels = np.concatenate([data_row["labels"] for data_row in data_rows])
        all_labels_names = np.array([INV_EVENT_ID[label] for label in all_labels])
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
                "label": all_labels,
                "label_name": all_labels_names,
                "epoch": all_epochs,
            }
        )

        torch.save(all_data, RAW_TENSORS_PATH / f"data/{split_type}_data.pt")
        np.save(RAW_TENSORS_PATH / f"labels/{split_type}_labels.npy", all_labels)

        classes = np.array(list(EVENT_ID.keys()))
        np.save(RAW_TENSORS_PATH / f"labels/classes.npy", classes)

        epochs_info.to_csv(SLEEP_EDF_PATH / f"{split_type}_epochs_info.csv", index=False)


def load_data_split(split_type):
    data = torch.load(RAW_TENSORS_PATH / "data" / f"{split_type}_data.pt")
    labels = np.load(RAW_TENSORS_PATH / "labels" / f"{split_type}_labels.npy")
    return data, labels
