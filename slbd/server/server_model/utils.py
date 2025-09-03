import asyncio
from typing import Union
from functools import partial

import numpy as np
import torch

from slbd.common import BatchDataForward


def split_numpy(data, split_sizes: Union[int, np.ndarray]):
    _size = split_sizes if isinstance(split_sizes, int) else split_sizes[:-1]
    if isinstance(data, np.ndarray):
        return np.split(data, _size, axis=0)
    if isinstance(data, list):
        tmp = [np.split(d, _size, axis=0) for d in data]
        return [list(t) for t in zip(*tmp)]
    else:
        raise ValueError(f"Unsupported type {type(data)}")


def _torch_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def pytorch_format(func):
    def wrapper(self, **kwargs):
        assert torch is not None
        assert len(kwargs) > 0, "No arguments provided to the function"
        torch_kwargs = {}
        sizes = None
        for kw, values in kwargs.items():
            tmp = [torch.from_numpy(v) for v in values]

            if sizes is None:
                sizes = [v.shape[0] for v in values]
            else:
                assert sizes == [v.shape[0] for v in values]

            torch_kwargs[kw] = torch.cat(tmp, dim=0)

        res = func(self, **torch_kwargs)

        cum_sizes = np.cumsum(sizes)
        split_fn = partial(split_numpy, split_sizes=cum_sizes)

        if res is None or len(res) == 0:
            # assume that there is a streaming request
            # TODO: how do enforce that methods invoked with streaming requests return None?
            return None

        if isinstance(res, torch.Tensor):
            return split_fn(_torch_to_numpy(res))
        elif isinstance(res, list):
            return split_fn([_torch_to_numpy(v) for v in res])

        assert isinstance(res, dict)

        out_list = [{} for _ in range(len(sizes))]
        for key, value in res.items():
            if isinstance(value, torch.Tensor):
                split_value = split_fn(_torch_to_numpy(value))
            elif isinstance(value, list):
                split_value = split_fn([_torch_to_numpy(v) for v in value])
            else:
                raise ValueError(f"Unsupported type {type(value)}")

            for i, v in enumerate(split_value):
                out_list[i][key] = v

        return out_list
    return wrapper


def single_client_format(func):
    def wrapper(self, **kwargs):
        assert all(len(v) == 1 for v in kwargs.values())
        kwargs = {k: v[0] for k, v in kwargs.items()}
        return [func(self, **kwargs),]
    return wrapper


async def set_event_after(event: asyncio.Event, delay: float):
    await asyncio.sleep(delay)
    if not event.is_set():
        event.set()


class ClientRequestGroup:
    """Container class that contains all the multiple requests from one or more clients.
    A ClientRequestGroup is a collection of client requests that:
    1. are meant to be processed by the same server model
    2. trigger the same method on the server model (either all for training or for evaluation)
    3. offload the same part of the computational graph to the server (i.e., have the
    same `num_offloaded_layers`). Thus, a ClientRequestGroup is a collection of requests
    that are meant to be processed together.
    """
    def __init__(self, num_offloaded_layers, server_id, timeout):
        self.server_id = server_id
        self.num_offloaded_layers = num_offloaded_layers
        self.batches = []
        self.events = []
        self.client_ids = [] # required for sorting the batches
        self.is_training = None
        self._is_ready = asyncio.Event()
        self._background_task = asyncio.create_task(set_event_after(self._is_ready, timeout))

    def add(self, batch_data: BatchDataForward, event, client_id: str):
        if self.is_training is not None:
            assert self.is_training == batch_data.is_training
        else:
            self.is_training = batch_data.is_training

        self.batches.append(batch_data)
        self.events.append(event)
        self.client_ids.append(client_id)

    def mark_as_ready(self):
        self._is_ready.set()

    def get_num_batches(self) -> int:
        return len(self.client_ids)

    def get_data(self):
        if not self._background_task.done():
            self._background_task.cancel()
        # sort by client_id to get consistent order
        sorted_indices = sorted(range(len(self.client_ids)), key=lambda k: self.client_ids[k])
        batches = [self.batches[idx] for idx in sorted_indices]
        events = [self.events[idx] for idx in sorted_indices]
        client_ids = [self.client_ids[idx] for idx in sorted_indices]

        return batches, events, client_ids
