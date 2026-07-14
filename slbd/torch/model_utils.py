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
    # If model is on meta device -> materialize it
    if next(model.parameters()).device.type == "meta":
        submodel = submodel.to_empty(device="cpu")
    return submodel


def extract_submodel_weights(
    model: FlexibleSequential,
    start_idx: int,
    end_idx: int,
    optimizer,
    extract_optimizer_states: bool,
) -> List[np.ndarray]:
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

    weights = [tensor for layer in layers_to_copy for _, tensor in iterate_params_and_state_dict(layer)]
    optimizer_states = []
    if extract_optimizer_states:
        # Keep this list positionally aligned with `weights`: buffers (e.g. batchnorm
        # running stats) and parameters that have not taken an optimizer step yet have
        # no optimizer state, so emit an empty dict rather than skipping them. The
        # empty placeholder is what lets the receiver zip the two lists together.
        for w in weights:
            state = optimizer.state.get(w, {})
            optimizer_states.append({
                k: v.detach() for k, v in state.items()
            })
    weights = [w.detach() for w in weights]
    return weights, optimizer_states


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


def param_list_to_state_dict(
    param_list: List[np.ndarray],
    model: FlexibleSequential
):
    if isinstance(param_list[0], np.ndarray):
        return OrderedDict([
            (key, torch.from_numpy(param)) for param, key in zip(param_list, model.state_dict().keys())
        ])
    else:
        return OrderedDict([
            (key, param) for param, key in zip(param_list, model.state_dict().keys())
        ])


def remove_layers_from_model(
    model: nn.Sequential,
    optimizer: torch.optim.Optimizer,
    remove_from: int,
) -> Tuple[FlexibleSequential, List[torch.Tensor], Optional[List[torch.Tensor]]]:
    # Remove layers from the model
    layers_to_keep = list(model.children())[:remove_from]
    layers_to_remove = list(model.children())[remove_from:]
    model = FlexibleSequential(*layers_to_keep)

    # Remove optimizer state for parameters that are no longer used
    for l in layers_to_remove:
        for p in l.parameters():
            if p in optimizer.state:
                del optimizer.state[p]

    return model


def concat_sequential_models(
    model1: FlexibleSequential,
    model2: FlexibleSequential,
    optimizer: Optional[torch.optim.Optimizer]
) -> FlexibleSequential:
    layers1 = list(model1.children())
    layers2 = list(model2.children())
    assert optimizer is not None
    if optimizer is not None:
        assert len(optimizer.param_groups) == 1
        optimizer.param_groups[0]["params"].extend(list(model2.parameters()))
    combined_layers = layers1 + layers2
    return FlexibleSequential(*combined_layers)


def iterate_params_and_state_dict(model):
    state_dict = model.state_dict()
    named_parameters = dict(model.named_parameters())
    for k, v in state_dict.items():
        if k in named_parameters:
            yield (k, named_parameters[k])
        else:
            yield (k, v)


def attach_layers(
    whole_model,
    current_model,
    num_new_layers,
    params_list: List[np.ndarray],
    optimizer,
    optimizer_states_list: Optional[List[np.ndarray]]=None
):
    # Get the device where current_model is located
    device = next(current_model.parameters()).device

    num_current_layers = get_num_layers(current_model)
    end_idx = num_current_layers + num_new_layers
    attach_model = extract_submodel(whole_model, num_current_layers, end_idx)

    if params_list is not None:
        state_dict = param_list_to_state_dict(params_list, attach_model)
        attach_model.load_state_dict(state_dict, strict=True)

    attach_model.to(device)
    attach_model.train(current_model.training)
    concat_model = concat_sequential_models(current_model, attach_model, optimizer)

    # Restore optimizer states for the new parameters if provided
    if optimizer_states_list:
        # Iterate params+buffers so this list is aligned with the one produced by
        # `extract_submodel_weights` on the sender (which includes buffers as empty states).
        override_optimizer_state(
            optimizer,
            optimizer_states_list,
            [param for _, param in iterate_params_and_state_dict(attach_model)],
        )
    concat_model._is_complete = end_idx == get_num_layers(whole_model)
    return concat_model


def override_optimizer_state(optimizer, optimizer_state_list, parameters):
    assert len(parameters) == len(optimizer_state_list), f"parameters {len(parameters)} optimizer {len(optimizer_state_list)}"
    for param, state in zip(parameters, optimizer_state_list):
        if not state:
            # empty state -> a buffer or a parameter that never took a step; nothing to restore
            continue
        optimizer.state[param] = {k: v.to(param.device) for k, v in state.items()}


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
