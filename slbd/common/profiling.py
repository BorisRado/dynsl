import os
import copy
from typing import Tuple, List

import torch
from fvcore.nn import FlopCountAnalysis

from slbd.common.utils import measure_time
from slbd.torch.sequential_model import FlexibleSequential


def _get_layer(model, optimizer, device, idx):
    layer = model[idx]
    layer.to(device)

    layer_requires_gradient = any(p.requires_grad for p in layer.parameters())
    if layer_requires_gradient:
        layer_optimizer = optimizer.__class__(layer.parameters(), **optimizer.defaults)
    else:
        layer_optimizer = None
    return layer, layer_optimizer

def _measure_layer_time_cv(layer, layer_optimizer, shape, device):
    x = torch.randn(shape, device=device)
    with measure_time(device.type == "cuda") as timer:
        preds = layer(x)
    forward_time = timer.elapsed

    if layer_optimizer is None:
        return forward_time, 0., preds

    # on the last layer we should consider also loss computation,
    # but we assume that the grad wrt logits is already computed for simplicity
    grad = torch.randn_like(preds)
    with measure_time(device.type == "cuda") as timer:
        layer_optimizer.zero_grad()
        preds.backward(grad)
        # Check if layer has any parameters that require gradients
        layer_optimizer.step()
    backward_time = timer.elapsed
    return forward_time, backward_time, preds


def _measure_per_layer_time_cv(
    model, batch_size, optimizer, device, iterations, dry_runs,
):
    forward_times = []
    backward_times = []
    output_tensor_floats = []
    next_layer_shape = (batch_size, 3, 32, 32)
    for layer_idx in range(len(model)):
        layer, layer_optimizer = _get_layer(model, optimizer, device, layer_idx)

        layer_forward_time = 0.
        layer_backward_time = 0.
        for iter_idx in range(iterations):
            iter_forward_time, iter_backward_time, preds = \
                _measure_layer_time_cv(layer, layer_optimizer, next_layer_shape, device)
            if iter_idx >= dry_runs:
                layer_forward_time += iter_forward_time
                layer_backward_time += iter_backward_time

        next_layer_shape = preds.shape
        output_tensor_floats.append(preds.numel())

        forward_times.append(layer_forward_time / (iterations - dry_runs))
        backward_times.append(layer_backward_time / (iterations - dry_runs))
    return forward_times, backward_times, output_tensor_floats


def _measure_time_embedding_layer(model, layer_optimizer, batch_size, sequence_length, device):
    token_ids = torch.randint(0, 1000, (batch_size, sequence_length), device=device)
    attention_mask = torch.ones(batch_size, sequence_length, device=device)
    with measure_time(device.type == "cuda") as timer:
        embeddings, extended_attention_mask = model[0](token_ids, attention_mask)
    forward_time = timer.elapsed
    grad = torch.randn_like(embeddings)

    with measure_time(device.type == "cuda") as timer:
        embeddings.backward(grad)
        layer_optimizer.step()
    backward_time = timer.elapsed
    return forward_time, backward_time, embeddings, extended_attention_mask


def _measure_time_transformer_block():
    pass



def _measure_per_layer_time_nlp(
    model, batch_size, optimizer, device, iterations, dry_runs,
):
    forward_times = []
    backward_times = []
    sequence_length = model.model_config.max_position_embeddings

    layer_forward_time = 0.
    layer_backward_time = 0.

    layer, layer_optimizer = _get_layer(model, optimizer, device, 0)
    for iter_idx in range(iterations):
        # treat the embedding layer separately
        iter_forward_time, iter_backward_time, embeddings, extended_attention_mask = \
            _measure_time_embedding_layer(
                model,
                layer_optimizer=layer_optimizer,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device
            )
        if iter_idx >= dry_runs:
            layer_forward_time += iter_forward_time
            layer_backward_time += iter_backward_time
    forward_times.append(layer_forward_time / (iterations - dry_runs))
    backward_times.append(layer_backward_time / (iterations - dry_runs))

    # we measure the time for every transformer layer, even though they are most likely equal
    shape = embeddings.shape
    for layer_idx in range(1, len(model)):
        layer_forward_time = 0.
        layer_backward_time = 0.
        layer, layer_optimizer = _get_layer(model, optimizer, device, layer_idx)

        for iter_idx in range(iterations):
            embeddings = torch.randn(shape, device=device)
            with measure_time(device.type == "cuda") as timer:
                if layer_idx < len(model) - 2:
                    preds = layer(embeddings, extended_attention_mask)
                else:
                    preds = layer(embeddings)
            if iter_idx >= dry_runs:
                layer_forward_time += timer.elapsed

            if layer_optimizer is None:
                continue
            # on the last layer we should consider also loss computation,
            # but we assume that the grad wrt logits is already computed for simplicity
            if isinstance(preds, tuple):
                grad = torch.randn_like(preds[0])
            else:
                grad = torch.randn_like(preds)
            with measure_time(device.type == "cuda") as timer:
                layer_optimizer.zero_grad()
                if isinstance(preds, tuple):
                    preds[0].backward(grad)
                else:
                    preds.backward(grad)

                # Check if layer has any parameters that require gradients
                layer_optimizer.step()
            if iter_idx >= dry_runs:
                layer_backward_time += timer.elapsed
        forward_times.append(layer_forward_time / (iterations - dry_runs))
        backward_times.append(layer_backward_time / (iterations - dry_runs))

    return forward_times, backward_times, []


def measure_per_layer_time(
    model: FlexibleSequential,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    iterations: int,
    dry_runs: int = 0,
) -> Tuple[List[float], List[float], List[int]]:
    """Measure the forward and backward time for every layer in the model

    Parameters
    ----------
    model : FlexibleSequential
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
    if model._data_type == "cv":
        return _measure_per_layer_time_cv(
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            device=device,
            iterations=iterations,
            dry_runs=dry_runs,
        )
    elif model._data_type == "nlp":
        return _measure_per_layer_time_nlp(
            model=model,
            batch_size=batch_size,
            optimizer=optimizer,
            device=device,
            iterations=iterations,
            dry_runs=dry_runs,
        )
    raise Exception("unknown model data type")


def measure_per_layer_complexity(disable_cuda=False, **kwargs):
    if torch.cuda.is_available() and disable_cuda:
        print("Measuring without cuda...")
        prev = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all GPUs
        torch.cuda.device_count()  # Force re-check of available devices

        forward_times, backward_times, tensor_sizes = measure_per_layer_time(**kwargs)

        if prev is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev
        torch.cuda.device_count()  # Reset state
    else:
        forward_times, backward_times, tensor_sizes = measure_per_layer_time(**kwargs)
    assert len(forward_times) == len(backward_times) == len(kwargs["model"])
    times = {
        i: fw+bw for i, (fw, bw) in enumerate(zip(forward_times, backward_times))
    }
    tensor_sizes = {
        i: ts for i, ts in enumerate(tensor_sizes)
    }
    return times, tensor_sizes

def get_flops_per_layer_fvcore(model):
    batch_size = 2
    if model._data_type == "cv":
        x = torch.randn(batch_size, 3, 32, 32)
    else:
        assert model._data_type == "nlp"
        sequence_length = model.model_config.max_position_embeddings
        token_ids = torch.randint(0, 1000, (batch_size, sequence_length))
        attention_mask = torch.ones(batch_size, sequence_length)
        x = (token_ids, attention_mask)

    flops = FlopCountAnalysis(model, x)
    flops_by_module = dict(flops.by_module())
    flops_by_module = {k: flops_by_module[str(k)] / batch_size for k in range(len(model))}
    return flops_by_module
