from typing import Union, Dict, List

import numpy as np
import torch

from flwr.common import ndarray_to_bytes, bytes_to_ndarray


def ndarray_dict_to_bytes(
    **data_dict
) -> Dict[str, Union[bytes, List[bytes]]]:

    out_dict = {}
    for key, val in data_dict.items():
        if isinstance(val, list):
            if isinstance(val[0], np.ndarray):
                out_dict[key] = [ndarray_to_bytes(v) for v in val]
            elif isinstance(val[0], dict):
                out_dict[key] = [{very_inner_key: ndarray_to_bytes(very_inner_val) for very_inner_key, very_inner_val in inner_val.items()} for inner_val in val]
            else:
                raise Exception(f"unknown type {type(val)}")
        else:
            out_dict[key] = ndarray_to_bytes(val)
    return out_dict


def bytes_to_ndarray_dict(**bytes_dict):
    out_dict = {}
    for key, val in bytes_dict.items():
        if isinstance(val, list):
            if isinstance(val[0], bytes):
                out_dict[key] = [bytes_to_ndarray(v) for v in val]
            elif isinstance(val[0], dict):
                out_dict[key] = [{very_inner_key: bytes_to_ndarray(very_inner_val) for very_inner_key, very_inner_val in inner_val.items()} for inner_val in val]
            else:
                raise Exception(f"unknown type {type(val[0])}")
        else:
            out_dict[key] = bytes_to_ndarray(val)
    return out_dict



def bytes_to_torch(bytes_data):
    out = bytes_to_ndarray_dict(bytes_data)
    if isinstance(out, np.ndarray):
        return torch.from_numpy(out)
    if isinstance(out, list):
        return [torch.from_numpy(o) for o in out]

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


def ndarray_list_to_bytes(ndarray_list):
    return [ndarray_to_bytes(v) for v in ndarray_list]
