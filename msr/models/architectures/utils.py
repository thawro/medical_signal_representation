from torch import nn


def _get_element(obj, i):
    return obj[i] if isinstance(obj, list) else obj


def _conv(dim: int, transpose: bool = False):
    transpose = "Transpose" if transpose else ""
    assert 1 <= dim <= 3
    return getattr(nn, f"Conv{transpose}{dim}d")


def _maxpool(dim: int):
    assert 1 <= dim <= 3
    return getattr(nn, f"MaxPool{dim}d")


def _batchnorm(dim: int):
    assert 1 <= dim <= 3
    return getattr(nn, f"BatchNorm{dim}d")
