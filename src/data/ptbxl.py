"""Module used to provide data for PTB-XL dataset"""

import ast
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import wfdb
from joblib import Parallel, delayed
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from data.utils import StratifiedBatchSampler
from signals.ecg import ECGSignal, create_multichannel_ecg
from signals.utils import parse_nested_feats

PTBXL_PATH = Path("./../../data/ptbxl")
RAW_DATASET_PATH = PTBXL_PATH / "raw"
RAW_TENSORS_DATA_PATH = PTBXL_PATH / "raw_tensors"
CHANNELS_POLARITY = [1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1]  # some channels are with oposite polarity


def load_raw_ptbxl_data(fs: float, target: str = "diagnostic_class"):
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


def load_data_and_save(fs: float = 100, target: str = "diagnostic_class", encode_labels=True):
    """Load PTB-XL data and save it

    Waveforms are saved as torch `.pt` files.
    Labels and classes mappings are saved as numpy  `.npy` files.

    Args:
        fs (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.
        encode_labels (bool): Whether to use `LabelEncoder` to encode labels. Defaults to `True`.
    """
    LABELS_PATH = RAW_TENSORS_DATA_PATH / f"labels/{target}"
    LABELS_PATH.mkdir(parents=True, exist_ok=True)

    DATA_PATH = RAW_TENSORS_DATA_PATH / f"records{fs}"
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    ptbxl_data = load_raw_ptbxl_data(fs, target)
    if encode_labels:
        label_encoder = LabelEncoder().fit(ptbxl_data["train"]["y"])
        ptbxl_data["train"]["y"] = label_encoder.transform(ptbxl_data["train"]["y"])
        ptbxl_data["val"]["y"] = label_encoder.transform(ptbxl_data["val"]["y"])
        ptbxl_data["test"]["y"] = label_encoder.transform(ptbxl_data["test"]["y"])
        np.save(LABELS_PATH / "classes.npy", label_encoder.classes_)

    for split in ["train", "val", "test"]:
        data_tensor = torch.tensor(ptbxl_data[split]["X"])
        labels_npy = ptbxl_data[split]["y"]
        torch.save(data_tensor, DATA_PATH / f"{split}_data.pt")
        # with open(, "wb") as f:
        np.save(LABELS_PATH / f"{split}_labels.npy", labels_npy)


def get_representations(channels_data: torch.Tensor, fs: float = 100) -> Dict[str, torch.Tensor]:
    """Get all types of representations (returned by ECGSignal objects).

    Args:
        channels_data (torch.Tensor): ECG channels data of shape `[TODO]`, where TODO.
        fs (float): Sampling rate. Defaults to 100.

    Returns:
        Dict[str, torch.Tensor]: Dict with representations names as keys and `torch.Tensor` objects as values.
    """
    multi_ecg = create_multichannel_ecg(channels_data.T.numpy(), fs)

    n_beats = 10
    multi_ecg.get_beats(source_channel=2)
    whole_signal_waveforms = multi_ecg.get_waveform_representation()
    whole_signal_features = multi_ecg.get_whole_signal_features_representation()
    per_beat_waveforms = multi_ecg.get_per_beat_waveform_representation(n_beats=n_beats)
    per_beat_features = multi_ecg.get_per_beat_features_representation(n_beats=n_beats)
    # whole_signal_embeddings = multi_ecg.get_whole_signal_embeddings_representation()

    actual_beats = per_beat_waveforms.shape[1]
    n_more_beats = n_beats - actual_beats
    # print(f"{actual_beats} beats in signal. Padding {n_more_beats} beats")
    per_beat_waveforms = np.pad(per_beat_waveforms, ((0, 0), (0, n_more_beats), (0, 0)))  # padding beats with zeros
    per_beat_features = np.pad(per_beat_features, ((0, 0), (0, n_more_beats), (0, 0)))  # padding beats with zeros

    return {
        "whole_signal_waveforms": whole_signal_waveforms,
        "whole_signal_features": whole_signal_features,
        "per_beat_waveforms": per_beat_waveforms,
        "per_beat_features": per_beat_features,
        # 'whole_signal_embeddings': whole_signal_embeddings,
    }


def create_representations_dataset(
    splits: List[str] = ["train", "val", "test"], fs: float = 100, use_multiprocessing: bool = True
):
    """Create and save data files (`.pt`) for all representations.

    Args:
        splits (list): Split types to create representations data for. Defaults to ['train', 'val', 'test'].
        fs (float): Sampling frequency of signals. Defaults to 100.
        use_multiprocessing (bool): Whether to use `joblib` library for multiprocessing
            or run everything in simple `for` loop. Defaults to `True`.
    """

    REPRESENTATIONS_DATA_PATH = PTBXL_PATH / f"representations_{fs}"
    REPRESENTATIONS_DATA_PATH.mkdir(parents=True, exist_ok=True)

    for split in tqdm(splits):
        data = torch.load(RAW_TENSORS_DATA_PATH / f"records{fs}/{split}_data.pt")
        if use_multiprocessing:
            representations = Parallel(n_jobs=-1)(delayed(get_representations)(channels, fs) for channels in tqdm(data))
        else:
            representations = [get_representations(channels) for channels in tqdm(data)]
        representation_names = representations[0].keys()
        representations = {
            name: torch.tensor(np.array([rep[name] for rep in representations])) for name in representation_names
        }
        for name, representation_data in representations.items():
            path = REPRESENTATIONS_DATA_PATH / name / f"{split}_data.pt"
            torch.save(representation_data, path)


def load_ptbxl_split(representation_type: str, fs: float, target: str, split: str) -> Dict[str, np.ndarray]:
    """Load PTB-XL train, val or test split for specified representation type, target and sampling rate (fs).

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
    REPRESENTATIONS_DATA_PATH = PTBXL_PATH / f"representations_{fs}"
    data = torch.load(REPRESENTATIONS_DATA_PATH / f"{representation_type}/{split}_data.pt")
    labels = np.load(RAW_TENSORS_DATA_PATH / f"labels/{target}/{split}_labels.npy")
    classes = np.load(RAW_TENSORS_DATA_PATH / f"labels/{target}/classes.npy", allow_pickle=True)
    classes = {i: classes[i] for i in range(len(classes))}
    return {"data": data, "labels": labels, "classes": classes}


class PTBXLDataset(Dataset):
    """PTB-XL Dataset class used in DeepLearning models."""

    def __init__(self, representation_type, fs, target, split, transform=None):
        dataset = load_ptbxl_split(representation_type, fs, target, split)
        self.data = dataset["data"]
        self.labels = dataset["labels"]
        self.classes = dataset["classes"]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx].float()
        data = torch.nan_to_num(data, nan=0.0)  # TODO !!!
        return data, self.labels[idx]


class PTBXLDataModule(LightningDataModule):
    """PTB-XL DataModule class used as DeepLearning models DataLoaders provider."""

    def __init__(
        self,
        representation_type: str,
        fs: float = 100,
        target: str = "diagnostic_class",
        batch_size: int = 64,
        num_workers=8,
    ):
        super().__init__()
        self.representation_type = representation_type
        self.fs = fs
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = PTBXLDataset(
                self.representation_type,
                self.fs,
                self.target,
                split="train",
            )
            self.val = PTBXLDataset(self.representation_type, self.fs, self.target, split="val")
        if stage == "test" or stage is None:
            self.test = PTBXLDataset(self.representation_type, self.fs, self.target, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_sampler=StratifiedBatchSampler(self.train[:][1], batch_size=self.batch_size, shuffle=True),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_sampler=StratifiedBatchSampler(self.val[:][1], batch_size=10 * self.batch_size, shuffle=False),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_sampler=StratifiedBatchSampler(self.test[:][1], batch_size=10 * self.batch_size, shuffle=False),
            num_workers=self.num_workers,
        )
