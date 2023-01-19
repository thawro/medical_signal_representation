import inspect
import json
import os
import zipfile
from collections.abc import Iterable
from itertools import groupby
from pathlib import Path
from typing import Callable, List, Union

import hydra
import numpy as np
import pandas as pd
import pyrootutils
import rich
import rich.syntax
import rich.tree
import torch
import wget
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

ROOT = Path(ROOT)
CONFIG_PATH = ROOT / "configs"
DATA_PATH = ROOT / "data"

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

EPSILON = np.finfo(float).eps

datamodule2str = {  # dm
    "PtbXLDataModule": "ptbxl",
    "MimicDataModule": "mimic",
    "MimicCleanDataModule": "mimic_clean",
    "SleepEDFDataModule": "sleep_edf",
}

model2str = {
    "LogisticRegression": "Regression",
    "LinearRegression": "Regression",
    "DecisionTreeClassifier": "Decision Tree",
    "DecisionTreeRegressor": "Decision Tree",
    "RandomForestClassifier": "Random Forest",
    "RandomForestRegressor": "Random Forest",
    "LGBMClassifier": "LGBM",
    "LGBMRegressor": "LGBM",
    "MLPClassifier": "MLP",
    "MLPRegressor": "MLP",
    "CNNClassifier": "CNN",
    "CNNRegressor": "CNN",
    "LSTMClassifier": "LSTM",
    "LSTMRegressor": "LSTM",
}


def dataset_to_str(project):
    datamodule = project.split(".")[-1]
    return datamodule2str[datamodule]


def logger_name_to_str(name):
    dataset, rep_type, model_target = name.split("__")
    model_classname = model_target.split(".")[-1]
    return f"{dataset_to_str(dataset)}__{rep_type}__{model_classname}"


def get_corr_matrix(df: pd.DataFrame):
    """Return correlation matrix using data from passed DataFrame.

    :func:`np.corrcoef` function is used and the `abs` values are passed.

    Args:
        df (pd.DataFrame): Source of data for correlation matrix.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    return pd.DataFrame(abs(np.corrcoef(df, rowvar=False)), columns=df.columns, index=df.columns)


def print_info(msg, verbose=0):
    if verbose > 0:
        print(msg)


def append_txt_to_file(filename: str, txt: Union[str, List[str]]):
    with open(filename, "a") as myfile:
        if isinstance(txt, list):
            for line in txt:
                myfile.write(f"{line}\n")
        else:
            myfile.write(f"{txt}\n")


def find_true_intervals(arr):
    return [
        (idxs[0], idxs[-1])
        for idxs in (list(group) for key, group in groupby(range(len(arr)), key=arr.__getitem__) if key)
    ]


def download_zip_and_extract(zip_file_url, dest_path):
    zip_file_path = str(dest_path / "tmp.zip")

    log.info(f"Downloading zip file to {zip_file_path}.")
    wget.download(zip_file_url, zip_file_path)

    log.info(f"Unzipping {zip_file_path} file.")
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(dest_path)

    log.info(f"Removing {zip_file_path} file.")
    os.remove(zip_file_path)

    old_dir_name = zip_file_url.split("/")[-1].split(".zip")[0]
    old_dir, new_dir = str(dest_path / old_dir_name), str(dest_path / "raw")
    log.info(f"Renaming {old_dir} to {new_dir}.")
    os.rename(old_dir, new_dir)


def run_in_parallel_with_joblib(fn: Callable, iterable: Iterable, n_jobs: int = -1, desc=None):
    return Parallel(n_jobs=n_jobs)(delayed(fn)(*args) for args in tqdm(iterable, total=len(iterable), desc=desc))


def create_new_obj(obj: object, **kwargs) -> object:
    """Create new instance of object passed with new arguments specified in `kwargs`.

    Args:
        obj (object): Object used as source of attributes for new object instance.

    Returns:
        object: New object instance with newly specified attributed.
    """
    init_args = list(inspect.signature(obj.__init__).parameters.items())
    args = {}
    for arg, val in init_args:
        if arg in kwargs.keys():
            args[arg] = kwargs[arg]
        else:
            if arg in vars(obj).keys():
                args[arg] = getattr(obj, arg)
            else:
                args[arg] = val.default
    return obj.__class__(**args)


def lazy_property(fn: Callable) -> Callable:
    """Lazily evaluated property.

    Args:
        fn (Callable): Function to decorate.

    Returns:
        Callable: Decorated function.
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


def load_tensor(path):
    return torch.load(path)


def ordered_dict_to_dict(dct):
    """Parse nested OrderedDict to dict"""
    try:
        return json.loads(json.dumps(dct))
    except TypeError:
        return dict(dct)


def print_config_tree(cfg: DictConfig, keys: List[str] = "all", style: str = "dim"):
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)
    if keys == "all":
        branch = tree.add("config", style=style, guide_style=style)
        branch_content = OmegaConf.to_yaml(cfg, resolve=True)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    else:
        for key, group in cfg.items():
            if key in keys:
                branch = tree.add(key, style=style, guide_style=style)
                branch_content = OmegaConf.to_yaml(group, resolve=True)
                branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)


def align_left(name, value, n_signs=15, sign=" = "):
    return f"{f'{name}':<{n_signs}}{sign}{value}"
