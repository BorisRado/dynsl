from enum import Enum

import torch.nn as nn
import torch.multiprocessing as mp


from slbd.server.server_model.torch_server_model import TorchServerModel


class TaskType(Enum):
    TRAIN = 0
    EVALUATE = 1
    GET_WEIGHTS = 2
    CONFIGURE_FIT = 3
    CONFIGURE_EVALUATE = 4
    GET_FIT_RESULT = 5
    SHUTDOWN = 6


class ModelWorker(mp.Process):
    def __init__(
        self,
        model: nn.Sequential,
        device,
        request_queue=None,
        response_queue=None,
        connection=None,
    ):
        super().__init__()
        self.model = TorchServerModel(model, device)
        if connection is not None:
            self.conn = connection
            assert request_queue is None and response_queue is None
            self.receive_fn = lambda: self.conn.recv()
            self.send_fn = lambda data: self.conn.send(data)

        else:
            assert connection is None
            self.request_queue = request_queue
            self.response_queue = response_queue
            self.receive_fn = lambda: self.request_queue.get()
            self.send_fn = lambda data: self.response_queue.put(data)

        self.device = device

    def run(self):
        while True:
            # First, check for control messages
            control, payload = self.receive_fn()
            if control == TaskType.TRAIN:
                embeddings, labels, num_offloaded_layers = payload
                gradient, training_loss = \
                    self.model.train_and_compute_gradient(
                        embeddings.to(self.device),
                        labels.to(self.device),
                        num_offloaded_layers
                    )
                data = (gradient, training_loss)
                self.send_fn(data)

            elif control == TaskType.EVALUATE:
                embeddings, num_offloaded_layers = payload
                logits = self.model.eval_get_logits(
                    embeddings.to(self.device),
                    num_offloaded_layers
                )
                self.send_fn(logits)

            elif control == TaskType.CONFIGURE_FIT:
                fit_ins = payload
                self.model.configure_fit(fit_ins)
                del fit_ins
                self.send_fn(None)

            elif control == TaskType.CONFIGURE_EVALUATE:
                evaluate_ins = payload
                self.model.configure_evaluate(evaluate_ins)
                del evaluate_ins
                self.send_fn(None)

            elif control == TaskType.GET_FIT_RESULT:
                print("Getting final results")
                self.send_fn(self.model.get_fit_result())

            elif control == TaskType.GET_WEIGHTS:
                raise Exception("Not implemented yet!")

            elif control == TaskType.SHUTDOWN:
                break
