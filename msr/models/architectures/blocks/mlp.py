from typing import List, Union

from torch import nn

from msr.models.architectures.helpers import (
    _activation,
    _batchnorm,
    get_ith_layer_kwargs,
)


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_norm: bool = False,
        batch_norm: bool = True,
        dropout: float = 0,
        activation: str = "ReLU",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_norm = weight_norm
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.activation = activation

        if weight_norm:
            layers = [nn.utils.weight_norm(nn.Linear(in_features, out_features))]
        else:
            layers = [nn.Linear(in_features, out_features)]
        if batch_norm:
            layers.append(_batchnorm(dim=1)(out_features))
        if activation is not None:
            layers.append(_activation(activation)())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FeedForward(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dims: List[int],
        weight_norm: Union[bool, List[bool]] = False,
        batch_norm: Union[bool, List[bool]] = True,
        dropout: Union[float, List[float]] = 0,
        activation: Union[str, List[str]] = "ReLU",
    ):
        super().__init__()
        hidden_dims = [input_size] + hidden_dims
        in_dims = hidden_dims[:-1]
        out_dims = hidden_dims[1:]
        n_layers = len(in_dims)
        layers = [
            FeedForwardBlock(
                **get_ith_layer_kwargs(
                    i,
                    in_features=in_dims,
                    out_features=out_dims,
                    weight_norm=weight_norm,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    activation=activation,
                )
            )
            for i in range(n_layers)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
