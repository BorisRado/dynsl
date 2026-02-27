import time
import uvloop
import asyncio
from functools import wraps
from collections import OrderedDict
from contextlib import contextmanager

import torch


def run_async(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Check if there's an active event loop
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop is running
            loop = None

        if loop and loop.is_running():
            # If there's a running event loop, call it as an async function
            return func(*args, **kwargs)

        # If no event loop is running, create one and run the function
        return uvloop.run(func(*args, **kwargs))

    return wrapper


def weighted_average_state_dicts(weighted_state_dicts):
    """
    Compute weighted average of state dicts.

    Args:
        weighted_state_dicts: List of tuples (weight, state_dict)

    Returns:
        OrderedDict containing weighted average of state dicts
    """
    if not weighted_state_dicts:
        return None

    # Get total weight
    total_weight = sum(weight for weight, _ in weighted_state_dicts)

    # Initialize averaged state dict with zeros
    result = OrderedDict()
    first_dict = weighted_state_dicts[0][1]
    for key in first_dict.keys():
        result[key] = torch.zeros_like(first_dict[key])

    # Compute weighted sum
    for weight, state_dict in weighted_state_dicts:
        for key in state_dict.keys():
            result[key] += weight * state_dict[key]

    # Divide by total weight
    for key in result.keys():
        result[key] = result[key] / total_weight

    return result


class TimerResult:
    def __init__(self):
        self.elapsed = None


@contextmanager
def measure_time(use_cuda):
    if use_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    timer = TimerResult()
    yield timer
    if use_cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()
    timer.elapsed = end - start


class MeasureTime:
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        self.elapsed = 0.

    def __enter__(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.use_cuda:
            torch.cuda.synchronize()
        tmp_elapsed = time.perf_counter() - self._start
        self.elapsed += tmp_elapsed

    def get_elapsed_and_reset(self):
        prev_elapsed = self.elapsed
        self.elapsed = 0.
        return prev_elapsed


class PerLayerTimer:
    """Efficient per-layer timing using CUDA events to avoid synchronization after each layer."""

    def __init__(self, use_cuda):
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.events = []
        else:
            self.times = []
            self._start_time = None

    def record_start(self):
        """Record the start of timing (before first layer)."""
        if self.use_cuda:
            self.events = []
            # Record initial event
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.events.append(start_event)
        else:
            self.times = []
            self._start_time = time.perf_counter()

    def record_layer_end(self):
        """Record the end of a layer (called after each layer forward)."""
        if self.use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self.events.append(end_event)
        else:
            elapsed = time.perf_counter() - self._start_time
            self.times.append(elapsed)

    def get_per_layer_times(self):
        """Get per-layer times. Synchronizes once for all CUDA events."""
        if self.use_cuda:
            # Synchronize once for all events
            torch.cuda.synchronize()

            layer_times = []
            for i in range(1, len(self.events)):
                elapsed_ms = self.events[0].elapsed_time(self.events[i])
                layer_times.append(elapsed_ms / 1000.0)  # Convert to seconds
            return layer_times
        else:
            return self.times
