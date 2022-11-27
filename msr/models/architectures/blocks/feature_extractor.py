from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, net: nn.Module, output_size: int):
        super().__init__()
        self.net = net
        self.output_size = output_size

    def forward(self, x):
        return self.net(x)
