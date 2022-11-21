import torch.nn.functional as F
from torch import nn

from msr.models.architectures.blocks import CNNBlock
from msr.models.architectures.utils import _get_element


class MLPExtractor(nn.Module):
    def __init__(self, in_dim=1000, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(nn.Linear(in_dim, num_classes))

    def forward(self, x):
        out = self.model(x)
        probs = F.softmax(out, dim=1)
        return probs


class LinearHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.net = nn.Linear(in_features, out_features)

    def forward(self, x):
        out = self.net(x)
        probs = F.softmax(out, dim=1)
        return probs


class CNNExtractor(nn.Module):
    def __init__(self, dim, in_channels, out_channels, maxpool_kernel_size, kernel_size, stride, padding, dilation):
        super().__init__()
        out_channels.insert(0, in_channels)
        self.net = nn.Sequential(
            *[
                CNNBlock(
                    dim=dim,
                    in_channels=_get_element(out_channels, i),
                    out_channels=_get_element(out_channels, i + 1),
                    maxpool_kernel_size=_get_element(maxpool_kernel_size, i),
                    kernel_size=_get_element(kernel_size, i),
                    stride=_get_element(stride, i),
                    padding=_get_element(padding, i),
                    dilation=_get_element(dilation, i),
                )
                for i in range(len(out_channels) - 1)
            ],
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.out_dim = out_channels[-1]

    def forward(self, x):
        out = self.net(x.permute(0, 2, 1))
        return out


class LSTMExtractor(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, bidirectional):
        super().__init__()
        output_layers = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.out_dim = output_layers * hidden_dim

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]


class NeuralNetwork(nn.Module):
    def __init__(self, feature_extractor, head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head
        self.num_classes = self.head.out_features

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.head(features)
