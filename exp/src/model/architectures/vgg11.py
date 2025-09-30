import torch
import torch.nn as nn

import torchvision.models as models

from slbd.torch.sequential_model import FlexibleSequential

from exp.src.common.stochasticity import set_seed


class VGG11(FlexibleSequential):
    def __init__(self, num_classes):
        model = models.vgg11(weights=None, num_classes=num_classes)
        all_layers = list(model.features.children()) + \
            [model.avgpool, nn.Flatten()] + \
            list(model.classifier.children())
        # Make sure no layer uses inplace operations
        for i, layer in enumerate(all_layers):
            if isinstance(layer, nn.ReLU) and layer.inplace:
                all_layers[i] = nn.ReLU(inplace=False)
        super().__init__(
            *all_layers
        )
        # self.meaningful_split_points = [idx+1 for idx, layer in enumerate(self.children()) if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d))]
        # self.meaningful_split_points = sorted(set(
        #     [point for i, point in enumerate(self.meaningful_split_points)
        #      if i == len(self.meaningful_split_points) - 1 or self.meaningful_split_points[i + 1] != point + 1]
        # )) + [len(self)]
        self.meaningful_split_points = [3, 6, 11, 16, 26, 30]


if __name__ == "__main__":
    set_seed(10)
    model = models.vgg11(weights=None, num_classes=10)

    print(sum(p.numel() for p in model.parameters()))
    set_seed(10)
    seq_model = VGG11(num_classes=10)

    x = torch.randn(10, 3, 32, 32)
    for _ in range(3):
        # just to warm up
        seq_model(x)

    import timeit
    start_time = timeit.default_timer()
    for _ in range(10):
        x = torch.randn(10, 3, 32, 32)
        model.eval()
        seq_model.eval()
        assert model(x).allclose(seq_model(x))
    print("Testing completed, alles gut", timeit.default_timer() - start_time)
