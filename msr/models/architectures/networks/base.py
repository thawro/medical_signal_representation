from torch import nn

from msr.models.architectures.blocks.feature_extractor import FeatureExtractor
from msr.models.architectures.blocks.head import (
    ClassificationHead,
    Head,
    RegressionHead,
)


class NeuralNetwork(nn.Module):
    def __init__(
        self, feature_extractor: FeatureExtractor, feed_forward: FeatureExtractor = None, head: Head = nn.Identity()
    ):
        super().__init__()
        if feed_forward is None:
            self.net = nn.Sequential(feature_extractor, head)
        else:
            self.net = nn.Sequential(feature_extractor, feed_forward, head)

    def forward(self, x):
        return self.net(x)


class RegressionNeuralNetwork(NeuralNetwork):
    def __init__(self, feature_extractor: FeatureExtractor, feed_forward: FeatureExtractor = None):
        output_size = feed_forward.output_size if feed_forward is not None else feature_extractor.output_size
        head = RegressionHead(in_features=output_size)
        super().__init__(feature_extractor=feature_extractor, feed_forward=feed_forward, head=head)


class ClassificationNeuralNetwork(NeuralNetwork):
    def __init__(
        self, feature_extractor: FeatureExtractor, feed_forward: FeatureExtractor = None, num_classes: int = 1
    ):
        output_size = feed_forward.output_size if feed_forward is not None else feature_extractor.output_size
        self.num_classes = num_classes
        head = ClassificationHead(in_features=output_size, num_classes=num_classes)
        super().__init__(feature_extractor=feature_extractor, feed_forward=feed_forward, head=head)
