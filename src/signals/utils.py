import inspect
from typing import Dict, Union

import pandas as pd


def create_new_obj(obj: Object, **kwargs) -> Object:
    """Create new instance of object passed with new arguments specified in `kwargs`.

    Args:
        obj (Object): Object used as source of attributes for new object instance.

    Returns:
        Object: New object instance with newly specified attributed.
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


def lazy_property(fn: function) -> function:
    """Lazily evaluated property.

    Args:
        fn (function): Function to decorate.

    Returns:
        function: Decorated function.
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


def parse_nested_feats(
    nested_feats: Dict[str, Dict[Union[float, int, str]]], sep: str = "__"
) -> Dict[str, Union[float, int, str]]:
    """Flatten nested features, i.e. keys are concatenated using specified separator.

    Args:
        nested_feats (Dict[str, Dict[Union[str, float]]]): Dict with nested keys.

    Returns:
        Dict[str, Union[float, int, str]]: Flattened dict.
    """
    feats_df = pd.json_normalize(nested_feats, sep=sep)
    feats = feats_df.to_dict(orient="records")[0]
    return {name[2:]: val for name, val in feats.items()}


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
