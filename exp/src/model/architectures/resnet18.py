import torch
import torch.nn as nn
import torchvision.models as models

from slbd.torch.sequential_model import FlexibleSequential

from exp.src.common.stochasticity import set_seed


def replace_bn_with_gn(module, num_groups=2):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(
                num_groups=num_groups,
                num_channels=child.num_features,
                affine=True
            )
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child, num_groups)


class ResNet18(FlexibleSequential):
    def __init__(self, num_classes, use_groupnorm):
        model = models.resnet18(weights=None, num_classes=num_classes)
        # replace the first conv layers, otherwise the resolution is reduced too much
        # same as in https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        if use_groupnorm:
            replace_bn_with_gn(model, num_groups=2) # 2 groups as in adaptive federated optimization
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
        self.meaningful_split_points = [4, 5, 6, 8]
        self._data_type = "cv"


if __name__ == "__main__":
    set_seed(10)
    model = models.resnet18(weights=None, num_classes=10)
    replace_bn_with_gn(model, num_groups=2)
    print(sum(p.numel() for p in model.parameters()))
    set_seed(10)
    seq_model = ResNet18(num_classes=10)

    for _ in range(10):
        x = torch.randn(10, 3, 224, 224)
        model.eval()
        seq_model.eval()
        assert model(x).allclose(seq_model(x))
    print("Testing completed, alles gut")
