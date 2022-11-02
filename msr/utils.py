from itertools import groupby
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data"


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
