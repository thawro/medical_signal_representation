from typing import List, Tuple, Union

from sorcery import dict_of
from torch import nn

from msr.models.architectures.blocks.cnn import CNN, ResidualCNN
from msr.models.architectures.blocks.feature_extractor import FeatureExtractor
from msr.models.architectures.blocks.other import BasicBlock1d, ResNet1d
from msr.models.architectures.helpers import _adaptive_pool
from msr.models.architectures.networks.base import (
    ClassificationNeuralNetwork,
    RegressionNeuralNetwork,
)
from msr.models.architectures.networks.mlp import MLPExtractor


class CNNExtractor(FeatureExtractor):
    def __init__(
        self,
        in_channels: int,
        conv0_kernel_size: int,
        conv0_channels: int,
        layers: List[int] = [1, 1, 1],
    ):
        cnn_net = ResNet1d(
            block=BasicBlock1d,
            kernel_size_stem=conv0_kernel_size,
            inplanes=conv0_channels,
            layers=layers,
            input_channels=in_channels,
        )
        net = nn.Sequential(
            cnn_net,
            _adaptive_pool(dim=1, mode="Avg")(1),
            nn.Flatten(),
        )
        output_size = conv0_channels
        super().__init__(net=net, output_size=output_size)

    def forward(self, x):
        out = self.net(x)
        return out


class CNNRegressor(RegressionNeuralNetwork):
    def __init__(
        self,
        in_channels: int,
        conv0_kernel_size: int,
        conv0_channels: int,
        layers: List[int] = [1, 1, 1],
        ff_hidden_dims: List[int] = [128, 128],
        ff_dropout: float = 0.2,
    ):
        feature_extractor = CNNExtractor(**dict_of(in_channels, conv0_kernel_size, conv0_channels, layers))
        feed_forward = MLPExtractor(
            input_size=feature_extractor.output_size, hidden_dims=ff_hidden_dims, dropout=ff_dropout
        )
        super().__init__(feature_extractor=feature_extractor, feed_forward=feed_forward)


class CNNClassifier(ClassificationNeuralNetwork):
    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        conv0_kernel_size: int,
        conv0_channels: int,
        layers: List[int] = [1, 1, 1],
        ff_hidden_dims: List[int] = [128, 128],
        ff_dropout: float = 0.2,
    ):
        feature_extractor = CNNExtractor(**dict_of(in_channels, conv0_kernel_size, conv0_channels, layers))
        feed_forward = MLPExtractor(
            input_size=feature_extractor.output_size, hidden_dims=ff_hidden_dims, dropout=ff_dropout
        )
        super().__init__(feature_extractor=feature_extractor, feed_forward=feed_forward, num_classes=num_classes)
