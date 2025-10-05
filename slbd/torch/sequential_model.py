import torch.nn as nn


class FlexibleSequential(nn.ModuleList):
    def __init__(self, *layers):
        super().__init__(layers)

    def forward(self, *inputs):
        if len(inputs) == 1:
            inputs = inputs[0]
            for layer in self.children():
                inputs = layer(inputs)
        else:
            for layer in self.children():
                inputs = layer(*inputs)
        return inputs
