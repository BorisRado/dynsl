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
    remove_from: int,
    extract_optimizer_states: bool = False
) -> Tuple[FlexibleSequential, List[torch.Tensor], Optional[List[torch.Tensor]]]:
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

    # 2. Extract optimizer states for removed parameters if requested
    removed_optimizer_states = None
    if extract_optimizer_states:
        removed_optimizer_states = []
        # Get parameters in a consistent order (layers order)
        ordered_remove_parameters = []
        for layer in layers_to_remove:
            ordered_remove_parameters.extend(list(layer.parameters()))

        for p in ordered_remove_parameters:
            if p in optimizer.state:
                # Extract optimizer state for this parameter
                state = optimizer.state[p]

                # Extract states in a consistent order based on optimizer type
                if isinstance(optimizer, torch.optim.SGD):
                    if 'momentum_buffer' in state:
                        removed_optimizer_states.append(state['momentum_buffer'].detach().cpu().numpy())
                    else:
                        # Add a placeholder if no momentum buffer exists
                        removed_optimizer_states.append(np.zeros_like(p.detach().cpu().numpy()))
                elif isinstance(optimizer, torch.optim.Adam):
                    if 'step' in state:
                        removed_optimizer_states.append(np.array([state['step']]))
                    else:
                        removed_optimizer_states.append(np.array([0]))

                    if 'exp_avg' in state:
                        removed_optimizer_states.append(state['exp_avg'].detach().cpu().numpy())
                    else:
                        removed_optimizer_states.append(np.zeros_like(p.detach().cpu().numpy()))

                    if 'exp_avg_sq' in state:
                        removed_optimizer_states.append(state['exp_avg_sq'].detach().cpu().numpy())
                    else:
                        removed_optimizer_states.append(np.zeros_like(p.detach().cpu().numpy()))
            else:
                # Parameter has no optimizer state, add placeholders
                if isinstance(optimizer, torch.optim.SGD):
                    removed_optimizer_states.append(np.zeros_like(p.detach().cpu().numpy()))
                elif isinstance(optimizer, torch.optim.Adam):
                    removed_optimizer_states.append(np.array([0]))  # step
                    removed_optimizer_states.append(np.zeros_like(p.detach().cpu().numpy()))  # exp_avg
                    removed_optimizer_states.append(np.zeros_like(p.detach().cpu().numpy()))  # exp_avg_sq

    # 3. Remove optimizer state for parameters that are no longer used
    for p in remove_parameters:
        if p in optimizer.state:
            del optimizer.state[p]
    # 4. Filter parameter groups as well
    for group in optimizer.param_groups:
        group['params'] = [p for p in group['params'] if p not in remove_parameters]

    return model, removed_parameters, removed_optimizer_states


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


def attach_layers(whole_model, current_model, num_new_layers, params_list, optimizer, optimizer_states_list=None):
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
    if optimizer_states_list is not None:
        _restore_optimizer_states(optimizer, attach_model.parameters(), optimizer_states_list)

    concat_model._is_complete = end_idx == get_num_layers(whole_model)
    return concat_model


def get_state_dict_keys_between_limits(model, start_idx, end_idx):
    return [
        key for key in model.state_dict().keys()
        if start_idx <= int(key.split('.')[0]) < end_idx
    ]


def get_layer_idx(name):
    return int(name.split(".")[0])


def _restore_optimizer_states(optimizer, parameters, optimizer_states_list):
    """
    Restore optimizer states for the given parameters from a flattened list of states.

    Args:
        optimizer: The optimizer to restore states for
        parameters: The parameters to restore states for (should be in layer order)
        optimizer_states_list: Flattened list of optimizer state tensors in consistent order
    """
    if not optimizer_states_list:
        return

    state_idx = 0
    # Convert parameters to list to ensure consistent ordering
    param_list = list(parameters) if not isinstance(parameters, list) else parameters

    for param in param_list:
        if state_idx >= len(optimizer_states_list):
            break

        # Initialize optimizer state for this parameter if it doesn't exist
        if param not in optimizer.state:
            optimizer.state[param] = {}

        state = optimizer.state[param]

        # Restore states in the same order they were extracted
        if isinstance(optimizer, torch.optim.SGD):
            # Only momentum_buffer for SGD
            if state_idx < len(optimizer_states_list):
                momentum_tensor = torch.from_numpy(optimizer_states_list[state_idx])
                # Only set momentum_buffer if it's not all zeros (wasn't a placeholder)
                if not np.allclose(momentum_tensor.numpy(), 0):
                    state['momentum_buffer'] = momentum_tensor.to(param.device)
                state_idx += 1
        elif isinstance(optimizer, torch.optim.Adam):
            # Adam: step (scalar), exp_avg, exp_avg_sq in that order
            if state_idx < len(optimizer_states_list):
                step_val = int(optimizer_states_list[state_idx][0])
                if step_val > 0:  # Only set if not a placeholder
                    state['step'] = step_val
                state_idx += 1
            if state_idx < len(optimizer_states_list):
                exp_avg_tensor = torch.from_numpy(optimizer_states_list[state_idx])
                if not np.allclose(exp_avg_tensor.numpy(), 0):  # Only set if not a placeholder
                    state['exp_avg'] = exp_avg_tensor.to(param.device)
                state_idx += 1
            if state_idx < len(optimizer_states_list):
                exp_avg_sq_tensor = torch.from_numpy(optimizer_states_list[state_idx])
                if not np.allclose(exp_avg_sq_tensor.numpy(), 0):  # Only set if not a placeholder
                    state['exp_avg_sq'] = exp_avg_sq_tensor.to(param.device)
                state_idx += 1


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
