import os
import copy
import time
from typing import Tuple, List

import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis

from slbd.common.utils import measure_time
from slbd.torch.sequential_model import FlexibleSequential
from slbd.torch.model_utils import extract_submodel


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
    device: torch.device,
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
    device = torch.device("cuda" if torch.cuda.is_available() and not disable_cuda else "cpu")
    forward_times, backward_times, tensor_sizes = measure_per_layer_time(**kwargs, device=device)
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


def register_time():
    if torch.cuda.is_available():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    else:
        return time.perf_counter()


class PerLayerTimer:

    def __init__(self, model, optimizers):
        assert len(optimizers) == len(model)
        self.model = model
        self.optimizers = optimizers
        self.use_cuda = torch.cuda.is_available()


    def compute_difference(self, target, reference):
        if self.use_cuda:
            return target.elapsed_time(reference) / 1e3
        else:
            return reference - target

    def sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def __call__(self, input):
        # add hooks to the model -- MUST BE ADDED BEFORE FORWARD
        backward_timestamps, forward_timestamps, optimizer_timestamps = [], [], []

        x = input

        # forward pass
        forward_timestamps = [register_time()]
        for layer in self.model:
            x = layer(x)
            forward_timestamps.append(register_time())
            x.register_hook(lambda _: backward_timestamps.insert(0, register_time()))

        # backward pass
        grad = torch.randn_like(x).detach()
        backward_timestamps = [register_time()]
        x.backward(grad)

        # optimizer step
        optimizer_timestamps = [register_time()]
        for optim in self.optimizers:
            if optim is not None:
                optim.step()
            optimizer_timestamps.append(register_time())

        self.sync()

        # compute the forward & backward pass layer-by-layer
        forward_times = [self.compute_difference(prev, nxt) for prev, nxt in zip(forward_timestamps[:-1], forward_timestamps[1:])]
        backward_times = [self.compute_difference(nxt, prev) for prev, nxt in zip(backward_timestamps[:-1], backward_timestamps[1:])]
        optimizer_steps_times = [self.compute_difference(prev, nxt) for prev, nxt in zip(optimizer_timestamps[:-1], optimizer_timestamps[1:])]

        assert all([a >= 0. for a in forward_times])
        assert all([a >= 0. for a in backward_times])
        assert all([a >= 0. for a in optimizer_steps_times])
        per_layer_total_time = [sum(v) for v in zip(forward_times, backward_times, optimizer_steps_times)]
        return forward_times, backward_times, optimizer_steps_times, per_layer_total_time



def profile_cumulative_per_layer_time_full_model(model: torch.nn.Sequential, optimizer_name: str, batch_size: int, iterations, dry_runs):
    model = copy.deepcopy(model)
    if torch.cuda.is_available():
        model.cuda()

    optimizers = []
    all_forward_times, all_backward_times, all_optimizer_times, all_total_times = [], [], [], []
    for layer in model:
        if any(p.requires_grad for p in layer.parameters()):
            if optimizer_name.lower() == "adam":
                optimizers.append(torch.optim.Adam(layer.parameters()))
            elif optimizer_name.lower() == "sgd":
                optimizers.append(torch.optim.SGD(layer.parameters(), momentum=0.9))
            else:
                raise Exception("Unknown optimizer")

        else:
            optimizers.append(None)

    per_layer_timer = PerLayerTimer(model, optimizers)

    for iter_idx in range(iterations):
        x = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
        if torch.cuda.is_available():
            x = x.cuda()

        # create a new optimizer per layer so as to be able to measure the per-layer optimizer time
        fwd_times, backward_times, optimizer_times, per_layer_time = per_layer_timer(x)
        if iter_idx < dry_runs:
            continue
        all_forward_times.append(fwd_times)
        all_backward_times.append(backward_times)
        all_optimizer_times.append(optimizer_times)
        all_total_times.append(per_layer_time)

    assert len(all_forward_times) == iterations - dry_runs
    return np.cumsum(np.mean(all_total_times, axis=0)).tolist()


def profile_cumulative_per_layer_time_partial_model(model, optimizer_name, batch_size, iterations, dry_runs, considered_split_points=None):
    if considered_split_points is None:
        considered_split_points = list(range(1, len(model)+1))

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    times = []
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    for split_point in considered_split_points:
        submodel = extract_submodel(model, 0, split_point).to(device)
        assert len(submodel) == split_point
        assert optimizer_name == "adam"
        suboptimizer = torch.optim.Adam(submodel.parameters())

        grad = None
        for _ in range(dry_runs):
            suboptimizer.zero_grad()
            embeddings = submodel(x)
            if grad is None:
                grad = torch.randn_like(embeddings)
            embeddings.backward(grad)
            suboptimizer.step()

        start = register_time()
        for _ in range(iterations - dry_runs):
            suboptimizer.zero_grad()
            embeddings = submodel(x)
            embeddings.backward(grad)
            suboptimizer.step()

        end = register_time()
        if is_cuda:
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3 / (iterations - dry_runs))
        else:
            times.append((time.perf_counter() - start) / (iterations - dry_runs))

    return times


def profile_cumulative_per_layer_time_individual_layers(model, batch_size, iterations, dry_runs):

    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    times = []

    model = copy.deepcopy(model)
    if next(iter(model.parameters())).device.type == "meta":
        model = model.to_empty(device="cpu")
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    for layer in model.children():
        layer.to(device)
        requires_grad = False
        if any(p.requires_grad for p in layer.parameters()):
            optimizer = torch.optim.Adam(layer.parameters())
            requires_grad = True

        # FIX: Ensure input tensor requires grad for the backward pass to work effectively
        # on the layer weights, or at least ensure the computation graph is built.
        # We need to detach x from previous layers to treat this layer individually,
        # otherwise backprop tries to go back to the start of the model.
        x = x.detach()

        grad = None
        for _ in range(dry_runs):
            # Regenerate input noise locally or reuse x?
            # Reusing x is safer for shape consistency, but we must ensure graph creation.
            y = layer(x)
            if grad is None:
                grad = torch.randn_like(y).detach()
            if requires_grad:
                optimizer.zero_grad()
                y.backward(grad)
                optimizer.step()

        start = register_time()
        for _ in range(iterations - dry_runs):
            y = layer(x)
            if requires_grad:
                optimizer.zero_grad()
                y.backward(grad)
                optimizer.step()
        end = register_time()

        if is_cuda:
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3 / (iterations - dry_runs))
        else:
            times.append((time.perf_counter() - start) / (iterations - dry_runs))

        # Pass the output to the next layer (detached from graph)
        # We run one last forward to get the 'valid' output for the next layer's shape
        x = y.clone().requires_grad_(True)

    return np.cumsum(np.array(times)).tolist()
