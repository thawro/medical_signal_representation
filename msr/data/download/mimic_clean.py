import random

import mat73
import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks
from tqdm.auto import tqdm

from msr.utils import DATA_PATH, load_tensor

N_SAMPLES = 1000
MAX_SAMPLES_PER_SUBJECT = 8
SEED = 42

FS = 125
DATASET_NAME = "mimic_clean"
DATASET_PATH = DATA_PATH / DATASET_NAME
LOGS_PATH = DATASET_PATH / "logs"
SPLIT_INFO_PATH = LOGS_PATH / "split_info.csv"

RAW_DATASET_PATH = DATASET_PATH / "raw"
RAW_TENSORS_PATH = DATASET_PATH / "raw_tensors"
RAW_TENSORS_DATA_PATH = RAW_TENSORS_PATH / "data"
RAW_TENSORS_TARGETS_PATH = RAW_TENSORS_PATH / "targets" / "sbp_dbp_avg"

for path in [RAW_DATASET_PATH, LOGS_PATH, RAW_TENSORS_DATA_PATH, RAW_TENSORS_TARGETS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


def load_raw_data():
    data = []

    for file_idx in tqdm(range(1, 5), desc="Loading raw data from .mat files"):
        fname = f"Part_{file_idx}"
        file_data = mat73.loadmat(RAW_DATASET_PATH / f"{fname}.mat")[fname]
        data.extend(file_data)
    return data


def sample_subjects(subjects_ids, max_samples_per_subject):
    unique_ids = np.unique(subjects_ids)
    chosen_samples = []
    for _id in unique_ids:
        subject_ids = set(np.where(subjects_ids == _id)[0])
        if max_samples_per_subject > len(subject_ids):
            subject_samples = list(subject_ids)
        else:
            subject_samples = random.sample(subject_ids, k=max_samples_per_subject)
        chosen_samples.extend(subject_samples)
    chosen_samples = np.array(sorted(chosen_samples))
    return chosen_samples


def prepare_mimic_clean_data(max_samples_per_subject=8, n_samples=1000, seed=42):
    random.seed(seed)

    # PPG, ABP, ECG
    raw_data = load_raw_data()
    data = np.concatenate([measurement.reshape(3, -1, n_samples) for measurement in raw_data], axis=1)
    data = torch.from_numpy(data)

    samples_per_subject = torch.tensor([measurement.shape[1] // n_samples for measurement in raw_data])
    subjects_ids = np.concatenate([[i] * samples_num for i, samples_num in enumerate(samples_per_subject)])

    sample_ids = sample_subjects(subjects_ids, max_samples_per_subject=max_samples_per_subject)

    subjects_ids = subjects_ids[sample_ids]
    data = data[:, sample_ids]

    abp = data[1].numpy()

    sbp = np.array(
        [_abp[find_peaks(_abp, height=_abp.mean())[0]].mean() for _abp in tqdm(abp, "Gathering sbp target values")]
    )
    dbp = np.array(
        [_abp[find_peaks(-_abp, height=-_abp.mean())[0]].mean() for _abp in tqdm(abp, "Gathering dbp target values")]
    )

    valid_mask = ~(np.isnan(sbp) | np.isnan(dbp))

    sample_ids = sample_ids[valid_mask]
    subjects_ids = subjects_ids[valid_mask]
    data = data[:, valid_mask]
    data = torch.index_select(data, 0, torch.tensor([0, 2]))  # PPG, ECG
    data = data.permute(1, 0, 2)  # [2, S, 1000] -> [S, 2, 1000]
    sbp = sbp[valid_mask]
    dbp = dbp[valid_mask]

    info = pd.DataFrame({"sample_id": sample_ids, "subject_id": subjects_ids, "sbp": sbp, "dbp": dbp})

    return data, info


def save_data_and_targets_to_files(data, splits_info):
    splits_info.to_csv(SPLIT_INFO_PATH)
    splits = splits_info["split"].unique()
    for split in tqdm(splits, desc="Saving split files"):
        split_mask = splits_info["split"].values == split
        split_data = data[split_mask]  # [S, 2, 1000]

        sbp = splits_info["sbp"].values[split_mask]
        dbp = splits_info["dbp"].values[split_mask]
        sbp_dbp_avg = torch.tensor(np.array([sbp, dbp])).T

        torch.save(split_data, RAW_TENSORS_DATA_PATH / f"{split}.pt")
        torch.save(sbp_dbp_avg, RAW_TENSORS_TARGETS_PATH / f"{split}.pt")


def load_mimic_raw_tensors_for_split(split: str):
    data = load_tensor(RAW_TENSORS_DATA_PATH / f"{split}.pt")
    targets = load_tensor(RAW_TENSORS_TARGETS_PATH / f"{split}.pt")
    return data, targets
