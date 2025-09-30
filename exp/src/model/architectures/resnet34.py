import torch
import torch.nn as nn

import torchvision.models as models

from exp.src.common.stochasticity import set_seed


class ResNet34(nn.Sequential):
    def __init__(self, num_classes):
        model = models.resnet34(weights=None, num_classes=num_classes)
        super().__init__(
            model.conv1,
            model.bn1,
            nn.ReLU(), # replace the default relu as it is inplace
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Flatten(),
            model.fc,
        )


if __name__ == "__main__":
    set_seed(10)
    model = models.resnet34(weights=None, num_classes=10)
    print(sum(p.numel() for p in model.parameters()))
    set_seed(10)
    seq_model = ResNet34(num_classes=10)

    import timeit
    start_time = timeit.default_timer()
    for _ in range(10):
        x = torch.randn(10, 3, 32, 32)
        model.eval()
        seq_model.eval()
        assert model(x).allclose(seq_model(x))
    print("Testing completed, alles gut", timeit.default_timer() - start_time)
