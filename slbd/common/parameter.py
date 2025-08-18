from typing import Union, Dict, List

import numpy as np
import torch

from flwr.common import ndarray_to_bytes, bytes_to_ndarray, NDArray, NDArrays


def ndarray_dict_to_bytes(
    ndarray_dict: Union[NDArray, NDArrays, Dict[str, Union[NDArray, NDArrays]]]
) -> Dict[str, Union[bytes, List[bytes]]]:
    if not isinstance(ndarray_dict, dict):
        ndarray_dict = {"": ndarray_dict}

    return {
        key: [ndarray_to_bytes(v) for v in val] if isinstance(val, list) else ndarray_to_bytes(val)
        for key, val in ndarray_dict.items()
    }


def bytes_to_ndarray_dict(bytes_dict):
    out = {
        key: [bytes_to_ndarray(v) for v in val] if isinstance(val, list) else bytes_to_ndarray(val)
        for key, val in bytes_dict.items()
    }
    if len(out) == 1 and "" in out:
        return out[""]
    return out


def bytes_to_torch(bytes_data):
    out = bytes_to_ndarray_dict(bytes_data)
    if isinstance(out, np.ndarray):
        return torch.from_numpy(out)

    torch_out = {}
    for key, val in out.items():
        torch_out[key] = torch.from_numpy(val) if isinstance(val, np.ndarray) else \
            [torch.from_numpy(v) for v in val]
    return torch_out


def torch_to_bytes(**data_dict):
    np_dict = {}
    for key, val in data_dict.items():
        if val is None:
            continue
        np_dict[key] = val.detach().cpu().numpy() \
            if isinstance(val, torch.Tensor) else [v.detach().cpu().numpy() for v in val]

    data = ndarray_dict_to_bytes(np_dict)
    return data


def torch_list_to_bytes(values):
    if values is None:
        return None
    return [ndarray_to_bytes(v.detach().cpu().numpy()) for v in values]
