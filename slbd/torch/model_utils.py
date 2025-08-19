from typing import Optional
from collections import OrderedDict

import copy
import torch
import torch.nn as nn


def extract_submodel(model, start_idx, end_idx):
    """Creates a copy of the model weights starting from start_idx and ending in
    """
    max_num_layers = get_num_layers(model)
    assert end_idx <= max_num_layers
    if end_idx == -1:
        end_idx = max_num_layers

    sub_layers = copy.deepcopy(list(model.children())[start_idx:end_idx])
    submodel = nn.Sequential(*sub_layers)
    submodel._is_complete = end_idx == max_num_layers
    return submodel


def extract_submodel_weights(model, start_idx, end_idx):
    submodel = extract_submodel(model, start_idx, end_idx)
    return [v.detach().cpu() for _, v in submodel.state_dict().items()]


def load_submodel(model, start_idx, end_idx, state_dict):
    submodel = extract_submodel(model, start_idx, end_idx)
    submodel.load_state_dict(state_dict)
    return submodel


def get_model_layer_keys(model):
    return {
        k: [v for v in model.state_dict() if int(v.split(".")[0]) == k]
        for k in range(len(model))
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
