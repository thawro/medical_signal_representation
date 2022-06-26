import numpy as np
import pandas as pd


def get_corr_matrix(df):
    return pd.DataFrame(abs(np.corrcoef(df, rowvar=False)), columns=df.columns, index=df.columns)
