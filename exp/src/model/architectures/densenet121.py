import torch
import torch.nn as nn
import torchvision.models as models

from slbd.torch.sequential_model import FlexibleSequential

from exp.src.common.stochasticity import set_seed
from exp.src.model.utils import replace_bn_with_gn


class DenseNet121(FlexibleSequential):
    def __init__(self, num_classes, use_groupnorm, num_groups):
        model = models.densenet121(weights=None, num_classes=num_classes)

        if use_groupnorm:
            assert num_groups > 1
            replace_bn_with_gn(model, num_groups=num_groups) # 2 groups in adaptive federated optimization

        # adaptations to make the model appropriate for 32x32 images
        model.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )

        # Remove max pooling
        model.features.pool0 = nn.Identity()

        params = list(list(model.features.children())) + \
            [nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), model.classifier]
        super().__init__(*params)
        self.meaningful_split_points = [4, 6, 8, 10, 11, 12]
        self._data_type = "cv"


if __name__ == "__main__":
    set_seed(10)
    model = models.densenet121(weights=None, num_classes=10)
    print(sum(p.numel() for p in model.parameters()))
    set_seed(10)
    seq_model = DenseNet121(num_classes=10)

    for _ in range(5):
        x = torch.randn(10, 3, 224, 224)
        model.eval()
        seq_model.eval()
        assert model(x).allclose(seq_model(x))

    for _ in range(5):
        x = torch.randn(10, 3, 32, 32)
        model.eval()
        seq_model.eval()
        assert model(x).allclose(seq_model(x))

    x = torch.randn(1, 3, 32, 32)
    for idx in range(len(seq_model)):
        submodel = FlexibleSequential(*list(seq_model.children())[:idx])
        print(idx, submodel(x).numel())

    print("======")
    from slbd.torch.model_utils import extract_submodel
    for idx in seq_model.meaningful_split_points:
        submodel = extract_submodel(seq_model, 0, idx)
        print(idx, submodel(x).numel())

    print("Testing completed, alles gut")
