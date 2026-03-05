import torch
import torch.nn as nn
import torchvision.models as models

from slbd.torch.sequential_model import FlexibleSequential

from exp.src.model.utils import replace_bn_with_gn


class MobileNet(FlexibleSequential):
    def __init__(self, num_classes, use_groupnorm):
        model = models.mobilenet_v2(num_classes=num_classes)
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        if use_groupnorm:
            replace_bn_with_gn(model) # 2 groups in adaptive federated optimization
        super().__init__(*list(model.features), nn.AdaptiveAvgPool2d(1), nn.Flatten(), model.classifier)
        self.meaningful_split_points = list(range(3, 18)) + [22,]
        self._data_type = "cv"


if __name__ == "__main__":
    from exp.src.common.stochasticity import set_seed
    set_seed(10)
    seq_model = MobileNet(num_classes=10, use_groupnorm=False)
    set_seed(10)
    efficientnet = models.mobilenet_v2(num_classes=10)
    efficientnet.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

    efficientnet.eval()
    seq_model.eval()
    for _ in range(4):
        x = torch.randn(2, 3, 32, 32)
        assert efficientnet(x).allclose(seq_model(x))

    print("Tests passed, alles gut")
