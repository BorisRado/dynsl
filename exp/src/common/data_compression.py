import torch


def to_f16(tensor: torch.Tensor, *args, **kwargs):
    return tensor.half()


def to_f32(tensor: torch.Tensor, *args, **kwargs):
    return tensor.float()
