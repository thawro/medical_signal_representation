from typing import List, Tuple, Union

from sorcery import dict_of
from torch import nn

from msr.models.architectures.blocks.cnn import CNN
from msr.models.architectures.blocks.feature_extractor import FeatureExtractor
from msr.models.architectures.helpers import _adaptive_pool
from msr.models.architectures.networks.base import (
    ClassificationNeuralNetwork,
    RegressionNeuralNetwork,
)


class CNNExtractor(FeatureExtractor):
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
        net = nn.Sequential(
            CNN(
                **dict_of(
                    dim,
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
            ),
            _adaptive_pool(dim, mode="Avg")(1),
            nn.Flatten(),
        )
        output_size = out_channels[-1]
        super().__init__(net=net, output_size=output_size)

    def forward(self, x):
        out = self.net(x)
        return out


class CNNRegressor(RegressionNeuralNetwork):
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
        feature_extractor = CNNExtractor(
            **dict_of(
                dim,
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
        super().__init__(feature_extractor=feature_extractor)


class CNNClassifier(ClassificationNeuralNetwork):
    def __init__(
        self,
        num_classes: int,
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
        feature_extractor = CNNExtractor(
            **dict_of(
                dim,
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
        super().__init__(feature_extractor=feature_extractor, num_classes=num_classes)
