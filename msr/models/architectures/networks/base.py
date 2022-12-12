from torch import nn

from msr.models.architectures.blocks.feature_extractor import FeatureExtractor
from msr.models.architectures.blocks.head import (
    ClassificationHead,
    Head,
    RegressionHead,
)


class NeuralNetwork(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, head: Head):
        super().__init__()
        self.net = nn.Sequential(feature_extractor, head)

    def forward(self, x):
        return self.net(x)


class RegressionNeuralNetwork(NeuralNetwork):
    def __init__(self, feature_extractor: FeatureExtractor):
        head = RegressionHead(in_features=feature_extractor.output_size)
        super().__init__(feature_extractor=feature_extractor, head=head)


class ClassificationNeuralNetwork(NeuralNetwork):
    def __init__(self, feature_extractor: FeatureExtractor, num_classes: int):
        self.num_classes = num_classes
        head = ClassificationHead(in_features=feature_extractor.output_size, num_classes=num_classes)
        super().__init__(feature_extractor=feature_extractor, head=head)
