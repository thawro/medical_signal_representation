import ast

import numpy as np
import pandas as pd
import wfdb


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

    Y["diagnostic_class"] = Y.scp_codes.apply(
        lambda codes: aggregate_class(codes, "diagnostic_class")
    )
    Y["diagnostic_subclass"] = Y.scp_codes.apply(
        lambda codes: aggregate_class(codes, "diagnostic_subclass")
    )

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
