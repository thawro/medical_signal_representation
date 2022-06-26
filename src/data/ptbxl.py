import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from signals.ecg import ECGSignal
from signals.utils import parse_nested_feats

DATASET_PATH = Path("./../../data/ptbxl")
CHANNELS_POLARITY = [1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, 1]  # some channels are with oposite polarity


def load_ptbxl_data(sampling_rate, path, target="diagnostic_class"):
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

    return {
        "train": {"X": X[train_mask], "y": Y[train_mask][target]},
        "val": {"X": X[val_mask], "y": Y[val_mask][target]},
        "test": {"X": X[test_mask], "y": Y[test_mask][target]},
    }


def get_feats_from_all_channels(channels, label, plot=False):
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


def create_split_features_df(ptbxl_data, split_type):
    X, y = ptbxl_data[split_type]["X"], ptbxl_data[split_type]["y"]
    feats_list = Parallel(n_jobs=-1)(
        delayed(get_feats_from_all_channels)(channels, label)
        for channels, label in tqdm(zip(X, y), total=len(X), desc=f"{split_type} split")
    )
    feats_list = [f for f in feats_list if f is not None]
    df = pd.DataFrame(feats_list)
    return df


def create_dataset(sampling_rate=100, target="diagnostic_class"):
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
