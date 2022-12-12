import torch.nn.functional as F
from torch import nn


class Head(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.net = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.net(x)


class RegressionHead(Head):
    def __init__(self, in_features):
        super().__init__(in_features=in_features, out_features=1)


class ClassificationHead(Head):
    def __init__(self, in_features, num_classes):
        super().__init__(in_features=in_features, out_features=num_classes)
        self.num_classes = self.out_features

    def forward(self, x):
        out = self.net(x)
        probs = F.softmax(out, dim=1)
        return probs
