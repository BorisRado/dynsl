import torch.nn as nn

from slbd.torch.sequential_model import FlexibleSequential


class VGG9(FlexibleSequential):

    def _make_convblock(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _make_fcblock(self, in_feat, out_feat):
        return nn.Sequential(
            nn.Linear(in_feat, out_feat),
            nn.BatchNorm1d(out_feat),
            nn.ReLU(inplace=True),
        )

    def __init__(self, num_classes, block1_ch=64, block2_ch=128, block3_ch=256, fc_units=256):
        super().__init__(
            self._make_convblock(3, block1_ch),
            self._make_convblock(block1_ch, block1_ch),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_convblock(block1_ch, block2_ch),
            self._make_convblock(block2_ch, block2_ch),
            nn.MaxPool2d(kernel_size=2, stride=2),

            self._make_convblock(block2_ch, block3_ch),
            self._make_convblock(block3_ch, block3_ch),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(start_dim=1),
            self._make_fcblock(4*4*block3_ch, fc_units),
            self._make_fcblock(fc_units, fc_units),
            nn.Linear(fc_units, num_classes)

        )
