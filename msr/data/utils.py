import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm.auto import tqdm


def create_group_train_val_test_split_idxs(
    groups, sample_ids, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42
):
    train_size = train_size / (train_size + val_size + test_size)
    val_size = val_size / (train_size + val_size + test_size)
    test_size = 1 - train_size - val_size
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, val_test_idx = next(gss.split(sample_ids, groups=groups))

    val_test_groups = groups[val_test_idx]
    val_test_sample_ids = sample_ids[val_test_idx]
    _val_size = val_size / (val_size + test_size)
    gss = GroupShuffleSplit(n_splits=1, train_size=_val_size, random_state=random_state)
    val_idx, test_idx = next(gss.split(val_test_sample_ids, groups=val_test_groups))

    train_sample_ids = sample_ids[train_idx]
    val_sample_ids = val_test_sample_ids[val_idx]
    test_sample_ids = val_test_sample_ids[test_idx]

    _sample_ids = sample_ids.tolist()
    train_idx = np.array([_sample_ids.index(element) for element in tqdm(train_sample_ids, desc="train idxs")])
    val_idx = np.array([_sample_ids.index(element) for element in tqdm(val_sample_ids, desc="val idxs")])
    test_idx = np.array([_sample_ids.index(element) for element in tqdm(test_sample_ids, desc="test idxs")])

    return train_idx, val_idx, test_idx


def create_train_val_test_split_info(groups, info, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    train_idx, val_idx, test_idx = create_group_train_val_test_split_idxs(
        groups=groups,
        sample_ids=info["sample_id"].values,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    train_info = info.iloc[train_idx]
    val_info = info.iloc[val_idx]
    test_info = info.iloc[test_idx]

    train_info.loc[:, "split"] = "train"
    val_info.loc[:, "split"] = "val"
    test_info.loc[:, "split"] = "test"

    return pd.concat([train_info, val_info, test_info])
