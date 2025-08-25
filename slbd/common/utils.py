import time
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
        return asyncio.run(func(*args, **kwargs))

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
    start = time.time()
    timer = TimerResult()
    yield timer
    if use_cuda:
        torch.cuda.synchronize()
    end = time.time()
    timer.elapsed = end - start
