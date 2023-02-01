from typing import List

from sorcery import dict_of
from torch import nn

from msr.models.architectures.blocks.feature_extractor import FeatureExtractor
from msr.models.architectures.networks.base import (
    ClassificationNeuralNetwork,
    RegressionNeuralNetwork,
)
from msr.models.architectures.networks.mlp import MLPExtractor


class LSTMExtractor(FeatureExtractor):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0, bidirectional: bool = True):
        output_layers = 2 if bidirectional else 1
        net = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        output_size = output_layers * hidden_dim
        super().__init__(net=net, output_size=output_size)

    def forward(self, x):
        out, _ = self.net(x)
        return out[:, -1, :]


class LSTMRegressor(RegressionNeuralNetwork):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0,
        bidirectional: bool = True,
        hidden_dims: List[int] = [128, 128],
    ):
        feature_extractor = LSTMExtractor(**dict_of(in_dim, hidden_dim, num_layers, dropout, bidirectional))
        feed_forward = MLPExtractor(input_size=feature_extractor.output_size, hidden_dims=hidden_dims, dropout=0.2)
        super().__init__(feature_extractor=feature_extractor, feed_forward=feed_forward)


class LSTMClassifier(ClassificationNeuralNetwork):
    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0,
        bidirectional: bool = True,
        hidden_dims: List[int] = [128, 128],
    ):
        feature_extractor = LSTMExtractor(**dict_of(in_dim, hidden_dim, num_layers, dropout, bidirectional))
        feed_forward = MLPExtractor(input_size=feature_extractor.output_size, hidden_dims=hidden_dims, dropout=0.2)
        super().__init__(feature_extractor=feature_extractor, feed_forward=feed_forward, num_classes=num_classes)
