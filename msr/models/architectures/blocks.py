from typing import Tuple, Union

from torch import nn

from msr.models.architectures.utils import _batchnorm, _conv, _maxpool


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
    ):
        super().__init__()
        self.net = nn.Sequential(
            _conv(dim, transpose)(in_channels, out_channels, kernel_size, stride, padding, dilation),
            _maxpool(dim)(maxpool_kernel_size) if maxpool_kernel_size is not None else nn.Identity(),
            _batchnorm(dim)(out_channels) if use_batchnorm else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
