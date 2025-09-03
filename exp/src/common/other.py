import torch


def set_torch_flags():
    # these flags help us to achieve better numerical stability across experiments and hence
    # help achieving more reproducible results
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_flush_denormal(True)
