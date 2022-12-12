from typing import Literal

from torch import nn


def get_element(obj, i):
    return obj[i] if isinstance(obj, list) else obj


def get_layer_kwargs(i, **kwargs):
    return {name: get_element(value, i) for name, value in kwargs.items()}


def _conv(dim: Literal[1, 2, 3], transpose: bool = False):
    transpose = "Transpose" if transpose else ""
    assert 1 <= dim <= 3
    return getattr(nn, f"Conv{transpose}{dim}d")


def _pool(dim: Literal[1, 2, 3], mode: Literal["Avg", "Max"] = "Max"):
    assert 1 <= dim <= 3
    return getattr(nn, f"{mode}Pool{dim}d")


def _batchnorm(dim: Literal[1, 2, 3]):
    assert 1 <= dim <= 3
    return getattr(nn, f"BatchNorm{dim}d")


def _adaptive_pool(dim: Literal[1, 2, 3], mode: Literal["Avg", "Max"] = "Avg"):
    assert 1 <= dim <= 3
    return getattr(nn, f"Adaptive{mode}Pool{dim}d")
