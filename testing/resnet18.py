import torch
import torch.nn as nn

import torchvision.models as models

from testing.utils import set_seed

class ResNet18(nn.Sequential):
    def __init__(self):
        model = models.resnet18(weights=None, num_classes=10)
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
    set_seed()
    model = models.resnet18(weights=None, num_classes=10)
    set_seed()
    seq_model = ResNet18()

    for _ in range(10):
        x = torch.randn(10, 3, 224, 224)
        model.eval()
        seq_model.eval()
        assert model(x).allclose(seq_model(x))
    print("Testing completed, alles gut")
