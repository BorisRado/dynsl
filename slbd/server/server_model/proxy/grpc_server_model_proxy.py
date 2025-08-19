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

    def __call__(self, embeddings, labels, local_computation_time, training=True):
        batch_data = torch_to_bytes(embeddings=embeddings, labels=labels)
        if self.request_queue is None:
            self._initialize_stream()

        ins = server_model_pb2.BatchDataForward(
            data=to_grpc_format(batch_data),
            control_code=control_code_to_proto(ControlCode.OK),
            training=training,
            local_computation_time=local_computation_time,
        )
        self.request_queue.put(ins)
        res = next(self.response_stream)
        num_layers_diff = res.layers_diff
        return bytes_to_torch(from_grpc_format(res.data)), LayerChangeInfo(
            layers_diff=num_layers_diff,
            additional_layers_parameters=None if num_layers_diff <= 0 \
                else [v.numpy() for v in bytes_to_torch(from_grpc_format(res.data))]
        )

    def close_stream(self):
        if self.request_queue is None:
            return
        self.request_queue.put(server_model_pb2.BatchDataForward(
            control_code=control_code_to_proto(ControlCode.DO_CLOSE_STREAM)
        ))
        res = next(self.response_stream)
        assert control_code_from_proto(res.control_code) == ControlCode.STREAM_CLOSED_OK
