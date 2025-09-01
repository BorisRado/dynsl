import torch
import torch.nn as nn

import torchvision.models as models

from exp.src.common.stochasticity import set_seed


class ResNet18(nn.Sequential):
    def __init__(self, num_classes):
        model = models.resnet18(weights=None, num_classes=num_classes)
        super().__init__(
            model.conv1,
            model.bn1,
            model.relu,
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
    model = models.resnet18(weights=None, num_classes=10)
    set_seed(10)
    seq_model = ResNet18(num_classes=10)

    for _ in range(10):
        x = torch.randn(10, 3, 224, 224)
        model.eval()
        seq_model.eval()
        assert model(x).allclose(seq_model(x))
    print("Testing completed, alles gut")
