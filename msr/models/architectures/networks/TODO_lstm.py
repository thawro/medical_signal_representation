from torch import nn


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
