from typing import List, Union

from sorcery import dict_of

from msr.models.architectures.blocks.feature_extractor import FeatureExtractor
from msr.models.architectures.blocks.mlp import FeedForward
from msr.models.architectures.networks.base import (
    ClassificationNeuralNetwork,
    RegressionNeuralNetwork,
)


class MLPExtractor(FeatureExtractor):
    def __init__(
        self,
        input_size: int,
        hidden_dims: List[int],
        weight_norm: Union[bool, List[bool]] = False,
        batch_norm: Union[bool, List[bool]] = True,
        dropout: Union[float, List[float]] = 0,
        activation: Union[str, List[str]] = "ReLU",
    ):

        net = FeedForward(**dict_of(input_size, hidden_dims, weight_norm, batch_norm, dropout, activation))
        output_size = hidden_dims[-1]
        super().__init__(net=net, output_size=output_size)


class MLPRegressor(RegressionNeuralNetwork):
    def __init__(
        self,
        input_size: int,
        hidden_dims: List[int],
        weight_norm: Union[bool, List[bool]] = False,
        batch_norm: Union[bool, List[bool]] = True,
        dropout: Union[float, List[float]] = 0,
        activation: Union[str, List[str]] = "ReLU",
    ):
        feature_extractor = MLPExtractor(
            **dict_of(input_size, hidden_dims, weight_norm, batch_norm, dropout, activation)
        )
        super().__init__(feature_extractor=feature_extractor)


class MLPClassifier(ClassificationNeuralNetwork):
    def __init__(
        self,
        num_classes: int,
        input_size: int,
        hidden_dims: List[int],
        weight_norm: Union[bool, List[bool]] = False,
        batch_norm: Union[bool, List[bool]] = True,
        dropout: Union[float, List[float]] = 0,
        activation: Union[str, List[str]] = "ReLU",
    ):
        feature_extractor = MLPExtractor(
            **dict_of(input_size, hidden_dims, weight_norm, batch_norm, dropout, activation)
        )
        super().__init__(feature_extractor=feature_extractor, num_classes=num_classes)
