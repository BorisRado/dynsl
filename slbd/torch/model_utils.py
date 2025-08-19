
from typing import Optional, Dict, List
from collections import OrderedDict
import copy

import torch
import torch.nn as nn


def extract_submodel(model: nn.Sequential, start_idx: int, end_idx: int) -> nn.Sequential:
    """
    Creates a copy of model layers from start_idx to end_idx.

    Args:
        model: The source sequential model
        start_idx: Starting layer index (inclusive)
        end_idx: Ending layer index (exclusive), use -1 for all remaining layers

    Returns:
        Sequential model containing the specified layers

    Raises:
        AssertionError: If end_idx exceeds the number of layers in the model
    """
    max_num_layers = get_num_layers(model)
    if end_idx == -1:
        end_idx = max_num_layers

    assert end_idx <= max_num_layers, (
        f"end_idx ({end_idx}) cannot exceed model layers ({max_num_layers})"
    )
    assert start_idx >= 0, f"start_idx must be non-negative, got {start_idx}"
    assert start_idx < end_idx, f"start_idx ({start_idx}) must be less than end_idx ({end_idx})"

    sub_layers = copy.deepcopy(list(model.children())[start_idx:end_idx])
    submodel = nn.Sequential(*sub_layers)
    submodel._is_complete = end_idx == max_num_layers
    return submodel


def extract_submodel_weights(model: nn.Sequential, start_idx: int, end_idx: int) -> List[torch.Tensor]:
    """
    Extract weights from a submodel defined by layer indices.

    Args:
        model: The source model
        start_idx: Starting layer index
        end_idx: Ending layer index

    Returns:
        List of tensors containing the model weights
    """
    submodel = extract_submodel(model, start_idx, end_idx)
    return [v.detach().cpu() for _, v in submodel.state_dict().items()]


def load_submodel(
    model: nn.Sequential,
    start_idx: int,
    end_idx: int,
    state_dict: OrderedDict
) -> nn.Sequential:
    """
    Load a submodel with given state dict.

    Args:
        model: The source model
        start_idx: Starting layer index
        end_idx: Ending layer index
        state_dict: State dictionary to load

    Returns:
        Loaded submodel
    """
    submodel = extract_submodel(model, start_idx, end_idx)
    submodel.load_state_dict(state_dict)
    return submodel


def get_model_layer_keys(model: nn.Sequential) -> Dict[int, List[str]]:
    """
    Get the parameter keys for each layer in the model.

    Args:
        model: The sequential model

    Returns:
        Dictionary mapping layer index to list of parameter keys
    """
    return {
        layer_idx: [
            param_key for param_key in model.state_dict()
            if int(param_key.split(".")[0]) == layer_idx
        ]
        for layer_idx in range(len(model))
    }


def shift_state_dict(state_dict, offset):
    return OrderedDict([
        (str(int(k.split(".")[0]) + offset) + "." + ".".join(k.split(".")[1:]), v)
        for k, v in state_dict.items()
    ])


def get_num_layers(model):
    return len(list(model.children()))


def param_list_to_state_dict(param_list, model):
    return OrderedDict([
        (key, torch.from_numpy(param)) for param, key in zip(param_list, model.state_dict().keys())
    ])


def concat_sequential_models(
    model1: nn.Sequential,
    model2: nn.Sequential,
    optimizer: Optional[torch.optim.Optimizer]
) -> nn.Sequential:
    layers1 = list(model1.children())
    layers2 = list(model2.children())
    if optimizer is not None:
        optimizer.add_param_group({"params": model2.parameters()})
    combined_layers = layers1 + layers2
    return nn.Sequential(*combined_layers)


def attach_layers(whole_model, current_model, num_new_layers, params_list, optimizer):
    num_current_layers = get_num_layers(current_model)
    end_idx = num_current_layers + num_new_layers
    attach_model = extract_submodel(whole_model, num_current_layers, end_idx)
    state_dict = param_list_to_state_dict(params_list, attach_model)
    state_dict = shift_state_dict(state_dict, num_current_layers)
    attach_model.load_state_dict(state_dict, strict=False)
    concat_model = concat_sequential_models(current_model, attach_model, optimizer)
    concat_model._is_complete = end_idx == get_num_layers(whole_model)
    return concat_model


def get_state_dict_keys_between_limits(model, start_idx, end_idx):
    return [
        key for key in model.state_dict().keys()
        if start_idx <= int(key.split('.')[0]) < end_idx
    ]
