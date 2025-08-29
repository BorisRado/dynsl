import copy
from typing import Tuple, List

import torch
import torch.nn as nn

from slbd.common.utils import measure_time


def measure_per_layer_time(
    model: nn.Sequential,
    optimizer: torch.optim.Optimizer,
    input_shape: Tuple[int, ...],
    iterations: int,
    dry_runs: int = 0,
) -> Tuple[List[float], List[float], List[int]]:
    """Measure the forward and backward time for every layer in the model

    Parameters
    ----------
    model : nn.Sequential
        Model to be profiled
    optimizer : torch.optim.Optimizer
        Base optimizer used during training
    input_shape : Tuple[int, ...]
        Shape of the input tensor, e.g., a batch of images
    iterations : int
        Number of times the measurement is to be run on every client
    dry_runs : int, optional
        If larger than 0, the first dry_runs iterations will be ignored (useful if considering
        GPU, as the first iterations may be slower than the others because of context
        initialization), by default 0

    Returns
    -------
    Tuple[List[float], List[float]]
        List of forward and backward times for every layer
    """
    assert iterations > dry_runs, \
        "The number of overall iterations must be larger than the number of iterations to be skipped"

    # copy the model and optimizer so that updates that happen
    # in the function do not get reflected outside
    model = copy.deepcopy(model)
    optimizer = copy.deepcopy(optimizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    forward_times = []
    backward_times = []
    output_tensor_floats = []

    next_layer_shape = input_shape
    for layer in model.children():
        layer.to(device)

        layer_requires_gradient = any(p.requires_grad for p in layer.parameters())
        # Create a new optimizer with the same configuration but only for this layer's parameters
        if layer_requires_gradient:
            layer_optimizer = optimizer.__class__(layer.parameters(), **optimizer.defaults)

        layer_forward_time = 0.
        layer_backward_time = 0.
        for iter_idx in range(iterations):
            x = torch.randn(next_layer_shape, device=device)
            with measure_time(is_cuda) as timer:
                preds = layer(x)
            if iter_idx >= dry_runs:
                layer_forward_time += timer.elapsed

            if not layer_requires_gradient:
                continue
            # on the last layer we should consider also loss computation,
            # but we assume that the grad wrt logits is already computed for simplicity
            grad = torch.randn_like(preds)
            with measure_time(is_cuda) as timer:
                layer_optimizer.zero_grad()
                preds.backward(grad)
                # Check if layer has any parameters that require gradients
                layer_optimizer.step()
            if iter_idx >= dry_runs:
                layer_backward_time += timer.elapsed

        next_layer_shape = preds.shape
        output_tensor_floats.append(preds.numel())

        forward_times.append(layer_forward_time / (iterations - dry_runs))
        backward_times.append(layer_backward_time / (iterations - dry_runs))
    return forward_times, backward_times, output_tensor_floats
