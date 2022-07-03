import inspect
from collections import OrderedDict
from typing import Callable, Dict, Union

import numpy as np
import pandas as pd
from scipy.integrate import simps


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


def parse_nested_feats(
    nested_feats: Dict[str, Dict[str, Union[float, int, str]]], sep: str = "__"
) -> Dict[str, Union[float, int, str]]:
    """Flatten nested features, i.e. keys are concatenated using specified separator.

    Args:
        nested_feats (Dict[str, Dict[Union[str, float]]]): Dict with nested keys.

    Returns:
        Dict[str, Union[float, int, str]]: Flattened dict.
    """
    feats_df = pd.json_normalize(nested_feats, sep=sep)
    feats = feats_df.to_dict(orient="records")[0]
    return OrderedDict({name[2:]: val for name, val in feats.items()})


def parse_feats_to_array(features):
    return np.array(list(parse_nested_feats(features).values()))


def get_outliers_mask(arr: np.ndarray, IQR_scale: float) -> np.ndarray:
    """Return outliers mask defined by IQR method.

    Args:
        arr (np.ndarray): Array with outliers.
        IQR_scale (float): Interquartile range scale.

    Returns:
        np.ndarray: Boolean mask, where False values indicate outliers.
    """
    return (arr < np.mean(arr) + IQR_scale * np.std(arr)) & (arr > np.mean(arr) - IQR_scale * np.std(arr))


def z_score(arr: np.ndarray) -> np.ndarray:
    """Return array with z_score normalization applied"""
    return (arr - np.mean(arr)) / np.std(arr)


def calculate_area(sig, fs, start, end, use_abs=True, method="simps"):
    abs_sig = abs(sig) if use_abs else np.copy(sig)
    abs_sig = abs_sig[start : end + 1]
    if method == "trapz":
        return np.trapz(abs_sig, dx=fs)
    elif method == "simps":
        return simps(abs_sig, dx=fs)


def calculate_energy(sig, start, end):
    return sum(sig[start : end + 1] ** 2)


def calculate_slope(t, sig, start, end):
    return (sig[end] - sig[start]) / (t[end] - t[start])
