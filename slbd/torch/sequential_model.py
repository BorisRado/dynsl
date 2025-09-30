import torch.nn as nn


class FlexibleSequential(nn.ModuleList):
    def __init__(self, *layers):
        super().__init__(layers)

    def forward(self, *inputs):
        for layer in self.children():
            if isinstance(inputs, tuple):
                inputs = layer(*inputs)
            else:
                inputs = layer(inputs)
        return inputs
