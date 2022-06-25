import inspect

import pandas as pd


def create_new_obj(obj, **kwargs):
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


def parse_nested_feats(nested_feats):
    feats_df = pd.json_normalize(nested_feats, sep="__")
    feats = feats_df.to_dict(orient="records")[0]
    return {name[2:]: val for name, val in feats.items()}


def get_outliers_mask(arr, IQR_scale):
    return (arr < np.mean(arr) + IQR_scale * np.std(arr)) & (arr > np.mean(arr) - IQR_scale * np.std(arr))


def z_score(arr):
    return (arr - arr.mean()) / arr.std()
