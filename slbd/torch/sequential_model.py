import torch.nn as nn


class FlexibleSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, *inputs):
        for layer in self.layers:
            if isinstance(inputs, tuple):
                inputs = layer(*inputs)
            else:
                inputs = layer(inputs)
        return inputs
