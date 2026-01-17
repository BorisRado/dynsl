from queue import Queue
from logging import DEBUG

from flwr.common import log

from slbd.common import bytes_to_torch, torch_to_bytes, LayerChangeInfo
from slbd.common.serde import (
    control_code_to_proto,
    control_code_from_proto,
    from_grpc_format,
    to_grpc_format,
    ControlCode,
)
from slbd.server.server_model.proxy.server_model_proxy import ServerModelProxy
from slbd.proto import server_model_pb2_grpc
from slbd.proto import server_model_pb2


# pylint: disable=no-member
class GrpcServerModelProxy(ServerModelProxy):

    def __init__(
        self,
        *,
        stub: server_model_pb2_grpc.ServerModelStub,
        cid: str
    ) -> None:
        super().__init__(cid=cid)
        log(DEBUG, "Initializing proxy for CID %s", cid)
        self.stub = stub
        self.request_queue = None
        self.response_stream = None

    def _initialize_stream(self):
        self.request_queue = Queue(maxsize=1)
        self.response_stream = self.stub.StreamRequest(iter(self.request_queue.get, None))
        self.request_queue.put(server_model_pb2.BatchDataForward(
            control_code=control_code_to_proto(ControlCode.INIT_STREAM),
            cid=self.cid,
        ))

    def unary_request(self, embeddings, labels, local_computation_time, training):
        batch_data = torch_to_bytes(embeddings=embeddings, labels=labels)
        ins = server_model_pb2.BatchDataForward(
            data=to_grpc_format(batch_data),
            control_code=control_code_to_proto(ControlCode.OK),
            training=training,
            local_computation_time=local_computation_time,
            cid=self.cid,
        )
        res = self.stub.UnaryRequest(ins)
        num_layers_diff = res.layers_diff
        gradient = bytes_to_torch(from_grpc_format(res.data))
        layer_change_info = LayerChangeInfo(
            layers_diff=num_layers_diff,
            additional_layers_parameters=None if num_layers_diff <= 0 \
                else [v.numpy() for v in bytes_to_torch({"": list(res.additional_layers_parameters.tensors)})],
            additional_optimizer_states=None if not hasattr(res, 'additional_optimizer_states') or res.additional_optimizer_states is None \
                else [v.numpy() for v in bytes_to_torch({"": list(res.additional_optimizer_states.tensors)})]
        )
        return gradient, layer_change_info, res.training_loss

    def __call__(self, embeddings, labels, local_computation_time, tot_local_time, client_model_weights, removed_optimizer_states, training=True):
        # return self.unary_request(embeddings, labels, local_computation_time, training)
        batch_data = torch_to_bytes(
            embeddings=embeddings,
            labels=labels,
            client_model_weights=client_model_weights,
            removed_optimizer_states=removed_optimizer_states,
        )
        if self.request_queue is None:
            self._initialize_stream()

        ins = server_model_pb2.BatchDataForward(
            data=to_grpc_format(batch_data),
            control_code=control_code_to_proto(ControlCode.OK),
            training=training,
            local_computation_time=local_computation_time,
            tot_local_time=tot_local_time,
        )
        self.request_queue.put(ins)
        res = next(self.response_stream)
        num_layers_diff = res.layers_diff
        gradient = bytes_to_torch(from_grpc_format(res.data))
        gradient, additional_optimizer_states = \
            gradient["grad"], gradient.get("optim_states", None)
        layer_change_info = LayerChangeInfo(
            layers_diff=num_layers_diff,
            additional_layers_parameters=None if num_layers_diff <= 0 \
                else [v.numpy() for v in bytes_to_torch({"": list(res.additional_layers_parameters.tensors)})],
            additional_optimizer_states=additional_optimizer_states,
        )
        return gradient, layer_change_info, res.training_loss

    def close_stream(self, error=False):
        if self.request_queue is None:
            return
        if error is False:
            self.request_queue.put(server_model_pb2.BatchDataForward(
                control_code=control_code_to_proto(ControlCode.DO_CLOSE_STREAM)
            ))
        else:
            self.request_queue.put(server_model_pb2.BatchDataForward(
                control_code=control_code_to_proto(ControlCode.ERROR_PROCESSING_STREAM)
            ))
        res = next(self.response_stream)
        assert control_code_from_proto(res.control_code) == ControlCode.STREAM_CLOSED_OK
