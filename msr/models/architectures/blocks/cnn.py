from typing import List, Tuple, Union

from sorcery import dict_of
from torch import nn

from msr.models.architectures.helpers import (
    _activation,
    _batchnorm,
    _conv,
    _pool,
    get_ith_layer_kwargs,
)


class CNNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        maxpool_kernel_size: Union[int, Tuple[int]],
        kernel_size: Union[int, Tuple[int]],
        stride: Union[int, Tuple[int]] = 1,
        padding: Union[int, Tuple[int]] = 0,
        dilation: Union[int, Tuple[int]] = 1,
        transpose: bool = False,
        use_batchnorm: bool = True,
        activation: str = "ReLU",
    ):
        super().__init__()
        layers = [_conv(dim, transpose)(in_channels, out_channels, kernel_size, stride, padding, dilation)]
        if maxpool_kernel_size is not None:
            layers.append(_pool(dim)(kernel_size=maxpool_kernel_size))
        if use_batchnorm:
            layers.append(_batchnorm(dim)(out_channels))
        if activation is not None:
            layers.append(_activation(activation)())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: List[int],
        maxpool_kernel_size: Union[int, List[int], Tuple[int], List[Tuple[int]]],
        kernel_size: Union[int, List[int], Tuple[int], List[Tuple[int]]],
        stride: Union[int, List[int], Tuple[int], List[Tuple[int]]] = 1,
        padding: Union[int, List[int], Tuple[int], List[Tuple[int]]] = 0,
        dilation: Union[int, List[int], Tuple[int], List[Tuple[int]]] = 1,
        transpose: Union[bool, List[bool]] = False,
        use_batchnorm: Union[bool, List[bool]] = True,
        activation: Union[str, List[str]] = "ReLU",
    ):
        super().__init__()
        hidden_channels = [in_channels] + out_channels
        in_channels = hidden_channels[:-1]
        out_channels = hidden_channels[1:]
        n_layers = len(in_channels)

        layers = [
            CNNBlock(
                dim=dim,
                **get_ith_layer_kwargs(
                    i,
                    **dict_of(
                        in_channels,
                        out_channels,
                        maxpool_kernel_size,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        transpose,
                        use_batchnorm,
                        activation,
                    )
                )
            )
            for i in range(n_layers)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
