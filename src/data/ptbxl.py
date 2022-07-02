"""Module used to provide data for PTB-XL dataset"""

import ast
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
import wfdb
from joblib import Parallel, delayed
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from signals.ecg import ECGSignal
from signals.utils import parse_nested_feats

DATASET_PATH = Path("./../../data/ptbxl")
TENSORS_DATA_PATH = DATASET_PATH / "tensors"
CHANNELS_POLARITY = [1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1]  # some channels are with oposite polarity


def load_ptbxl_data(sampling_rate: float, path: Path, target: str = "diagnostic_class", encode_labels: bool = True):
    """_summary_

    Args:
        sampling_rate (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        path (Path): Path to PTB-XL wfdb data.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.
        encode_labels (bool): Whether to use `LabelEncoder` to encode labels. Defaults to `True`.
    """

    def load_raw_ptbxl_data(df, sampling_rate, path):
        if sampling_rate == 100:
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
    Y = pd.read_csv(path / "ptbxl_database.csv", index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_ptbxl_data(Y, sampling_rate, path)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path / "scp_statements.csv", index_col=0)
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

    if encode_labels:
        label_encoder = LabelEncoder().fit(dataset["train"]["y"])
        dataset["train"]["y"] = label_encoder.transform(dataset["train"]["y"])
        dataset["val"]["y"] = label_encoder.transform(dataset["val"]["y"])
        dataset["test"]["y"] = label_encoder.transform(dataset["test"]["y"])

    return dataset


def save_as_torch_and_npy_files(sampling_rate=100, target="diagnostic_class"):
    """Load PTB-XL data and save it as torch (input) and npy (target) files.

    Args:
        sampling_rate (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.
    """
    ptbxl_data = load_ptbxl_data(sampling_rate=sampling_rate, path=DATASET_PATH, target=target)

    for split in ["train", "val", "test"]:
        data_tensor = torch.tensor(ptbxl_data[split]["X"])
        labels_npy = ptbxl_data[split]["y"]
        torch.save(data_tensor, f"../../data/ptbxl/tensors/records{sampling_rate}/{split}_data.pt")
        with open(f"../../data/ptbxl/tensors/records{sampling_rate}/{split}_labels.npy", "wb") as f:
            np.save(f, labels_npy)


def get_feats_from_all_channels(channels: np.ndarray, label: Union[str, int], plot: bool = False):
    """Extract features from all channels

    Args:
        channels (np.ndarray): Numpy array with channels data. Shape `[C, S]`, where `C` is number of channels and `S` is number of samples.
        label (Union[str, int]): Label of data sample.
        plot (bool): Whether to plot feature extraction. Defaults to `False`.

    Returns:
        _type_: _description_
    """
    try:
        all_channels_feats = {}
        for i, data in enumerate(channels.T):
            multiplier = CHANNELS_POLARITY[i]
            sig = ECGSignal("ecg", multiplier * data, 100)
            secg = sig.aggregate()
            feats = sig.extract_features(plot=plot)
            feats = parse_nested_feats(feats)
            all_channels_feats[i] = feats
        all_channels_feats = parse_nested_feats(all_channels_feats)
        all_channels_feats["label"] = label
        return all_channels_feats
    except Exception:
        pass


def create_split_features_df(ptbxl_data: Dict[str, Dict[str, np.ndarray]], split_type: str):
    """Create `DataFrame` with features for specific split.

    Args:
        ptbxl_data (Dict[str, Dict[str, np.ndarray]]): Dict with PTB-XL data.
        split_type (str): Type of split. One of `["train", "val", "test"]`.

    Returns:
        pd.DataFrame: DataFrame with features (and label as last column).
    """
    X, y = ptbxl_data[split_type]["X"], ptbxl_data[split_type]["y"]
    feats_list = Parallel(n_jobs=-1)(
        delayed(get_feats_from_all_channels)(channels, label)
        for channels, label in tqdm(zip(X, y), total=len(X), desc=f"{split_type} split")
    )
    feats_list = [f for f in feats_list if f is not None]
    df = pd.DataFrame(feats_list)
    return df


def create_features_dataset(sampling_rate: float = 100, target: str = "diagnostic_class"):
    """Create dataset (train, val and test splits) with features.

    Args:
        sampling_rate (float): Sampling rate of PTB-XL dataset. `100` or `500`.
        target (str): Target of PTB-XL data. `"diagnostic_class"` or `"diagnostic_subclass"`. Defaults to `"diagnostic_class"`.

    Returns:
        Dict[str, Dict[str, Union[pd.DataFrame, np.ndarray]]]: Dict with train, val and test data splits.
    """
    ptbxl_data = load_ptbxl_data(sampling_rate=sampling_rate, path=DATASET_PATH, target=target)
    train_df = create_split_features_df(ptbxl_data, split_type="train")
    val_df = create_split_features_df(ptbxl_data, split_type="val")
    test_df = create_split_features_df(ptbxl_data, split_type="test")

    label_encoder = LabelEncoder().fit(train_df["label"].values)
    train_df["label"] = label_encoder.transform(train_df["label"].values)
    val_df["label"] = label_encoder.transform(val_df["label"].values)
    test_df["label"] = label_encoder.transform(test_df["label"].values)

    return {
        "train": {"X": train_df.drop("label", axis=1), "y": train_df["label"].values},
        "val": {"X": val_df.drop("label", axis=1), "y": val_df["label"].values},
        "test": {"X": test_df.drop("label", axis=1), "y": test_df["label"].values},
    }


class PTBXLWaveformDataset(Dataset):
    """PTB-XL Dataset class used in DeepLearning models."""

    def __init__(self, split, sampling_rate=100, target="diagnostic_class", transform=None):
        # TODO: target is still not used -> only diagnostic_class labels are saved as tensors
        # TODO: sampling_rate works only for 100 -> 100 fs samples are saves as tensors

        self.data = torch.load(TENSORS_DATA_PATH / f"records{sampling_rate}/{split}_data.pt")
        self.labels = np.load(TENSORS_DATA_PATH / f"records{sampling_rate}/{split}_labels.npy", allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.data[idx][:, 0].float(), self.labels[idx]


class PTBXLWaveformDataModule(LightningDataModule):
    """PTB-XL DataModule class used as DeepLearning models DataLoaders provider."""

    def __init__(self, sampling_rate=100, target="diagnostic_class", batch_size: int = 64, num_workers=8):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.target = target
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = None
        self.val = None
        self.test = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = PTBXLWaveformDataset(split="train", sampling_rate=self.sampling_rate, target=self.target)
            self.val = PTBXLWaveformDataset(split="val", sampling_rate=self.sampling_rate, target=self.target)
        if stage == "test" or stage is None:
            self.test = PTBXLWaveformDataset(split="test", sampling_rate=self.sampling_rate, target=self.target)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=10 * self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=10 * self.batch_size, num_workers=self.num_workers)
