import ast
import logging

import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from msr.data.namespace import DATA_PATH

PTBXL_PATH = DATA_PATH / "ptbxl"
RAW_DATASET_PATH = PTBXL_PATH / "raw"
RAW_TENSORS_PATH = PTBXL_PATH / "raw_tensors"
RAW_TENSORS_DATA_PATH = RAW_TENSORS_PATH / "data"
TARGETS_PATH = RAW_TENSORS_PATH / "targets"
TARGET_PATH = None


def load_raw_ptbxl_data(fs: float, target: str):
    """Load raw PTB-XL data.

    Args:
        fs (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.
    """

    def load_raw_data(df, fs, path):
        files = df.filename_lr if fs == 100 else df.filename_hr
        data = [wfdb.rdsamp(str(path / f)) for f in files]
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

    dataset = {
        "train": {"X": X[train_mask], "y": Y[train_mask][target].values},
        "val": {"X": X[val_mask], "y": Y[val_mask][target].values},
        "test": {"X": X[test_mask], "y": Y[test_mask][target].values},
    }

    return dataset


def load_raw_ptbxl_data_and_save_to_tensors(fs: float, target: str, encode_labels: bool):
    """Load raw PTB-XL data (as downloaded from physionet) and save it to tensors.

    Waveforms are saved as torch `.pt` files.
    Labels and classes mappings are saved as numpy  `.npy` files.

    Args:
        fs (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.
        encode_labels (bool): Whether to use `LabelEncoder` to encode labels. Defaults to `True`.
    """
    TARGET_PATH = TARGETS_PATH / target
    TARGET_PATH.mkdir(parents=True, exist_ok=True)

    ptbxl_data = load_raw_ptbxl_data(fs, target)
    if encode_labels:
        label_encoder = LabelEncoder().fit(ptbxl_data["train"]["y"])
        ptbxl_data["train"]["y"] = label_encoder.transform(ptbxl_data["train"]["y"])
        ptbxl_data["val"]["y"] = label_encoder.transform(ptbxl_data["val"]["y"])
        ptbxl_data["test"]["y"] = label_encoder.transform(ptbxl_data["test"]["y"])
        np.save(TARGET_PATH / "classes.npy", label_encoder.classes_)

    for split in ["train", "val", "test"]:
        data_tensor = torch.tensor(ptbxl_data[split]["X"])
        targets_npy = ptbxl_data[split]["y"]
        torch.save(data_tensor, DATA_PATH / f"{split}.pt")
        np.save(TARGET_PATH / f"{split}.npy", targets_npy)


@hydra.main(version_base=None, config_path="../../configs/data", config_name="raw")
def main(cfg: DictConfig):
    cfg = cfg.ptbxl
    log.info(cfg)

    for path in [PTBXL_PATH, RAW_DATASET_PATH, RAW_TENSORS_PATH, RAW_TENSORS_DATA_PATH, TARGETS_PATH]:
        path.mkdir(parents=True, exist_ok=True)

    load_raw_ptbxl_data_and_save_to_tensors(fs=cfg.fs, target=cfg.target, encode_labels=cfg.encode_labels)


if __name__ == "__main__":
    main()


# TODO: add files download from web if set to True
# TODO: check if works
# TODO: Add docstrings
