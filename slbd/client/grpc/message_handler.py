from typing import Dict

import torch.nn as nn

from flwr.client.client import (
    maybe_call_evaluate,
    maybe_call_fit,
    maybe_call_get_parameters,
    maybe_call_get_properties,
)
from flwr.client.typing import ClientFn
from flwr.common import Context, Message, Scalar
from flwr.common.constant import MessageType, MessageTypeLegacy
from flwr.common.recordset_compat import (
    evaluateres_to_recordset,
    fitres_to_recordset,
    getparametersres_to_recordset,
    getpropertiesres_to_recordset,
    recordset_to_evaluateins,
    recordset_to_fitins,
    recordset_to_getparametersins,
    recordset_to_getpropertiesins,
)

from slbd.common.constants import CLIENT_ID_CONFIG_KEY
from slbd.proto.server_model_pb2_grpc import ServerModelStub
from slbd.server.server_model.proxy.grpc_server_model_proxy import GrpcServerModelProxy
from slbd.client.flower_numpy_client import FlowerNumpyClientAdapter


def handle_legacy_message_from_msgtype(
    client_fn: ClientFn,
    model: nn.Sequential,
    message: Message,
    context: Context,
    server_model_stub: ServerModelStub,
    config: Dict[str, Scalar],
) -> Message:
    """Handle legacy message in the inner most mod."""
    cid = str(message.metadata.partition_id)
    assert len(message.content.configs_records) == 1
    config_record = next(iter(message.content.configs_records.values()))

    server_cid = ""
    if CLIENT_ID_CONFIG_KEY in config_record:
        server_cid = config_record.pop(CLIENT_ID_CONFIG_KEY)

    server_model_proxy = GrpcServerModelProxy(stub=server_model_stub, cid=server_cid)

    client = FlowerNumpyClientAdapter(
        client_id=cid,
        torch_client_fn=client_fn,
        whole_model=model,
        server_model_proxy=server_model_proxy,
        config=config
    )
    client = client.to_client()

    client.set_context(context)

    message_type = message.metadata.message_type

    # Handle GetPropertiesIns
    if message_type == MessageTypeLegacy.GET_PROPERTIES:
        get_properties_res = maybe_call_get_properties(
            client=client,
            get_properties_ins=recordset_to_getpropertiesins(message.content),
        )
        out_recordset = getpropertiesres_to_recordset(get_properties_res)
    # Handle GetParametersIns
    elif message_type == MessageTypeLegacy.GET_PARAMETERS:
        get_parameters_res = maybe_call_get_parameters(
            client=client,
            get_parameters_ins=recordset_to_getparametersins(message.content),
        )
        out_recordset = getparametersres_to_recordset(
            get_parameters_res, keep_input=False
        )
    # Handle FitIns
    elif message_type == MessageType.TRAIN:
        fit_res = maybe_call_fit(
            client=client,
            fit_ins=recordset_to_fitins(message.content, keep_input=True),
        )
        out_recordset = fitres_to_recordset(fit_res, keep_input=False)
    # Handle EvaluateIns
    elif message_type == MessageType.EVALUATE:
        evaluate_res = maybe_call_evaluate(
            client=client,
            evaluate_ins=recordset_to_evaluateins(message.content, keep_input=True),
        )
        out_recordset = evaluateres_to_recordset(evaluate_res)
    else:
        raise ValueError(f"Invalid message type: {message_type}")

    server_model_proxy.close_stream()

    # Return Message
    return message.create_reply(out_recordset)
