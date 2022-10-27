from pathlib import Path

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
