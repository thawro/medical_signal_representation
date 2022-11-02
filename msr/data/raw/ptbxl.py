import ast
import logging
from typing import Dict, Literal, Union

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from msr.utils import DATA_PATH

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

DATASET_NAME = "ptbxl_2"
ZIP_FILE_URL = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2.zip"
DATASET_PATH = DATA_PATH / DATASET_NAME
RAW_DATASET_PATH = DATASET_PATH / "raw"
RAW_TENSORS_PATH = DATASET_PATH / "raw_tensors"
RAW_TENSORS_DATA_PATH = RAW_TENSORS_PATH / "data"
TARGETS_PATH = RAW_TENSORS_PATH / "targets"
TARGET_PATH = None

for path in [RAW_DATASET_PATH, RAW_TENSORS_DATA_PATH, TARGETS_PATH]:
    path.mkdir(parents=True, exist_ok=True)


def load_raw_ptbxl_data(
    fs: float, target: Literal["diagnostic_class", "diagnostic_subclass"]
) -> Dict[str, Dict[str, Union[pd.DataFrame, np.ndarray]]]:
    """Load raw PTB-XL data.

    Args:
        fs (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.

    Returns:
        Dict[str, Dict[str, Union[pd.DataFrame, np.ndarray]]]: Dictionary with dataset.
    """

    def load_raw_data(df, fs, path):
        files = df.filename_lr if fs == 100 else df.filename_hr
        data = [wfdb.rdsamp(str(path / f)) for f in tqdm(files, desc="Reading files with wfdb")]
        data = np.array([signal for signal, meta in data])
        return data

    def aggregate_class(y_dict, class_name):
        tmp = []
        for key, val in y_dict.items():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key][class_name])
            tmp = list(set(tmp))
            if len(tmp) == 1:
                diagnostic_class = tmp[0]
            else:
                diagnostic_class = np.nan
        return diagnostic_class

    log.info("Loading raw ptbxl data.")
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
    VAL_FOLD, TEST_FOLD = 9, 10

    target_mask = ~pd.isnull(Y[target]).values

    val_mask = (Y["strat_fold"] == VAL_FOLD).values & target_mask
    test_mask = (Y["strat_fold"] == TEST_FOLD).values & target_mask
    train_mask = ~(val_mask | test_mask) & target_mask

    return {
        "train": {"X": X[train_mask], "y": Y[train_mask][target].values},
        "val": {"X": X[val_mask], "y": Y[val_mask][target].values},
        "test": {"X": X[test_mask], "y": Y[test_mask][target].values},
    }


def create_raw_tensors_dataset(fs: float, target: str, encode_targets: bool):
    """Load raw PTB-XL data (as downloaded from physionet) and save it to tensors.

    Waveforms are saved as torch `.pt` files.
    Targets are saved as torch `.pt` files.
    Classes mappings are saved as numpy  `.npy` files.

    Args:
        fs (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.
        encode_targets (bool): Whether to use `LabelEncoder` to encode labels. Defaults to `True`.
    """
    splits = ["train", "val", "test"]
    TARGET_PATH = TARGETS_PATH / target
    TARGET_PATH.mkdir(parents=True, exist_ok=True)
    log.info(f"Using {target} target.")
    ptbxl_data = load_raw_ptbxl_data(fs, target)
    if encode_targets:
        log.info("Encoding targets.")
        label_encoder = LabelEncoder().fit(ptbxl_data["train"]["y"])
        for split in splits:
            ptbxl_data[split]["y"] = label_encoder.transform(ptbxl_data[split]["y"])
        np.save(TARGET_PATH / "classes.npy", label_encoder.classes_)

    log.info("Creating train/val/test tensors.")

    for split in tqdm(splits, desc="Splits"):
        data = torch.tensor(ptbxl_data[split]["X"])
        targets_npy = ptbxl_data[split]["y"]
        targets = torch.from_numpy(targets_npy)
        torch.save(data, RAW_TENSORS_DATA_PATH / f"{split}.pt")
        torch.save(targets, TARGET_PATH / f"{split}.pt")
