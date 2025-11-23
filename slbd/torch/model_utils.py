from typing import Optional, Dict, List, Tuple
from collections import OrderedDict
import copy

import numpy as np
import torch
import torch.nn as nn

from slbd.torch.sequential_model import FlexibleSequential


def extract_submodel(model: FlexibleSequential, start_idx: int, end_idx: int) -> FlexibleSequential:
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
    submodel = FlexibleSequential(*sub_layers)
    submodel._is_complete = end_idx == max_num_layers
    return submodel


def extract_submodel_weights(model: FlexibleSequential, start_idx: int, end_idx: int) -> List[np.ndarray]:
    """
    Extract weights from a submodel defined by layer indices.

    Args:
        model: The source model
        start_idx: Starting layer index
        end_idx: Ending layer index

    Returns:
        List of tensors containing the model weights
    """
    if end_idx == -1:
        end_idx = get_num_layers(model)
    layers_to_copy = list(model.children())[start_idx:end_idx]

    return [tensor.detach().cpu().numpy() for layer in layers_to_copy for tensor in layer.state_dict().values()]


def extract_submodel_state_dict(model: FlexibleSequential, start_idx: int, end_idx: int) -> List[torch.Tensor]:
    """
    Extract weights from a submodel defined by layer indices.

    Args:
        model: The source model
        start_idx: Starting layer index
        end_idx: Ending layer index

    Returns:
        List of tensors containing the model weights
    """
    return shift_state_dict(extract_submodel(model, start_idx, end_idx).state_dict(), start_idx)


def load_submodel(
    model: FlexibleSequential,
    start_idx: int,
    end_idx: int,
    state_dict: OrderedDict
) -> FlexibleSequential:
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


def get_model_layer_keys(model: FlexibleSequential) -> Dict[int, List[str]]:
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
    return len(model)


def param_list_to_state_dict(param_list, model):
    return OrderedDict([
        (key, torch.from_numpy(param)) for param, key in zip(param_list, model.state_dict().keys())
    ])


def remove_layers_from_model(
    model: nn.Sequential,
    optimizer: torch.optim.Optimizer,
    remove_from: int
) -> Tuple[FlexibleSequential, List[torch.Tensor]]:
    # Remove layers from the model
    layers_to_keep = list(model.children())[:remove_from]
    layers_to_remove = list(model.children())[remove_from:]
    removed_parameters = [
        tensor.detach().cpu().numpy() for layer in layers_to_remove
        for tensor in layer.state_dict().values()
    ]
    model = FlexibleSequential(*layers_to_keep)

    # 1. Get the parameter IDs that the new model keeps
    remove_parameters = {p for l in layers_to_remove for p in l.parameters()}

    # 2. Remove optimizer state for parameters that are no longer used
    for p in remove_parameters:
        if p in optimizer.state:
            del optimizer.state[p]
    # 3. Filter parameter groups as well
    for group in optimizer.param_groups:
        group['params'] = [p for p in group['params'] if p not in remove_parameters]

    return model, removed_parameters


def concat_sequential_models(
    model1: FlexibleSequential,
    model2: FlexibleSequential,
    optimizer: Optional[torch.optim.Optimizer]
) -> FlexibleSequential:
    layers1 = list(model1.children())
    layers2 = list(model2.children())
    if optimizer is not None:
        optimizer.add_param_group({"params": model2.parameters()})
    combined_layers = layers1 + layers2
    return FlexibleSequential(*combined_layers)


def attach_layers(whole_model, current_model, num_new_layers, params_list, optimizer):
    # Get the device where current_model is located
    device = next(current_model.parameters()).device

    num_current_layers = get_num_layers(current_model)
    end_idx = num_current_layers + num_new_layers
    attach_model = extract_submodel(whole_model, num_current_layers, end_idx)

    if params_list is not None:
        state_dict = param_list_to_state_dict(params_list, attach_model)
        attach_model.load_state_dict(state_dict, strict=True)

    attach_model.to(device)
    concat_model = concat_sequential_models(current_model, attach_model, optimizer)
    concat_model._is_complete = end_idx == get_num_layers(whole_model)
    return concat_model


def get_state_dict_keys_between_limits(model, start_idx, end_idx):
    return [
        key for key in model.state_dict().keys()
        if start_idx <= int(key.split('.')[0]) < end_idx
    ]


def get_layer_idx(name):
    return int(name.split(".")[0])


if __name__ == "__main__":
    from exp.src.model.architectures.cnn import SimpleCNN

    model = SimpleCNN(10)
    assert get_state_dict_keys_between_limits(model, 0, 3) == ["0.weight", "0.bias"]
    assert get_state_dict_keys_between_limits(model, 0, 4) == ["0.weight", "0.bias", "3.weight", "3.bias"]

    print(shift_state_dict(model.state_dict(), offset=2).keys())
    assert list(shift_state_dict(model.state_dict(), offset=2).keys()) == [
        "2.weight", "2.bias",
        "5.weight", "5.bias",
        "8.weight", "8.bias",
        "12.weight", "12.bias",
        "14.weight", "14.bias"
    ]

    assert get_num_layers(model) == 13

    submodel = extract_submodel(model, 0, 5)
    assert all(
        (v1 == v2).all() for v1, v2 in zip(model.state_dict().values(), submodel.state_dict().values())
    )

    # reinitialize the weights of submodel
    for layer in submodel:
        if hasattr(layer, 'weight') and layer.weight is not None:
            nn.init.normal_(layer.weight)
        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.normal_(layer.bias)
    # make sure that there is no longer any dependency between the two models
    assert all(
        (v1 != v2).all() for v1, v2 in zip(model.state_dict().values(), submodel.state_dict().values())
    )

    # check attacking of layers to existing model
    # ... let's attach 3 more layers to `submodel`
    optimizer = torch.optim.SGD(submodel.parameters())

    to_be_attached_model_weights = [
        torch.randn_like(v).numpy() for v in extract_submodel_weights(model, 5, 8)
    ]
    new_model = attach_layers(model, submodel, 3, to_be_attached_model_weights, optimizer)
    assert (new_model[6].weight == to_be_attached_model_weights[0]).all()
    assert (new_model[6].bias == to_be_attached_model_weights[1]).all()

    print(new_model)
    print("All tests run all right")
