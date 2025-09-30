from enum import Enum

import torch.nn as nn
import torch.multiprocessing as mp

from slbd.torch.sequential_model import FlexibleSequential
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
        model: FlexibleSequential,
        device,
        request_queue=None,
        response_queue=None,
        connection=None,
        shared_tensors=None
    ):
        super().__init__()
        self.model = TorchServerModel(model, device)
        if connection is not None:
            self.conn = connection
            assert request_queue is None and response_queue is None
            if shared_tensors is None:
                self.receive_fn = self.receive_fn_pipe
                self.send_fn = self.send_fn_pipe
            else:
                self.shared_tensors = shared_tensors
                self.receive_fn = self.receive_fn_pipe_shared_memory
                self.send_fn = self.send_fn_pipe_shared_memory
        else:
            assert connection is None
            self.request_queue = request_queue
            self.response_queue = response_queue
            self.receive_fn = self.receive_fn_queue
            self.send_fn = self.send_fn_queue

        self.device = device

    def receive_fn_pipe_shared_memory(self):
        control, payload = self.conn.recv()
        if control == TaskType.TRAIN:
            num_offloaded_layers = payload
            return control, (
                self.shared_tensors["embeddings"],
                self.shared_tensors["labels"],
                num_offloaded_layers
            )
        else:
            return (control, payload)

    def send_fn_pipe_shared_memory(self, data, task_type):
        if task_type == TaskType.TRAIN:
            gradient, training_loss = data
            self.shared_tensors["gradient"].copy_(gradient)
            self.conn.send(training_loss)
        else:
            return self.conn.send(data)

    def receive_fn_pipe(self):
        return self.conn.recv()

    def send_fn_pipe(self, data, task_type):
        return self.conn.send(data)

    def receive_fn_queue(self):
        return self.request_queue.get()

    def send_fn_queue(self, data, task_type):
        return self.response_queue.put(data)

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
                self.send_fn(data, TaskType.TRAIN)

            elif control == TaskType.EVALUATE:
                embeddings, num_offloaded_layers = payload
                logits = self.model.eval_get_logits(
                    embeddings.to(self.device),
                    num_offloaded_layers
                )
                self.send_fn(logits, TaskType.EVALUATE)

            elif control == TaskType.CONFIGURE_FIT:
                fit_ins = payload
                self.model.configure_fit(fit_ins)
                del fit_ins
                self.send_fn(None, TaskType.CONFIGURE_FIT)

            elif control == TaskType.CONFIGURE_EVALUATE:
                evaluate_ins = payload
                self.model.configure_evaluate(evaluate_ins)
                del evaluate_ins
                self.send_fn(None, TaskType.CONFIGURE_EVALUATE)

            elif control == TaskType.GET_FIT_RESULT:
                print("Getting final results")
                self.send_fn(self.model.get_fit_result(), TaskType.GET_FIT_RESULT)

            elif control == TaskType.GET_WEIGHTS:
                raise Exception("Not implemented yet!")

            elif control == TaskType.SHUTDOWN:
                break
