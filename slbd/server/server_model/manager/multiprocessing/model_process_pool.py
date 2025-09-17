from abc import ABC, abstractmethod

import torch
import torch.multiprocessing as mp

from slbd.server.server_model.manager.multiprocessing.model_worker import ModelWorker, TaskType


class ModelProcessPool(ABC):
    def __init__(self, model):
        self._model = model
        self._idle_processes = []
        self._sid_to_active_process_mapping = {}

    @abstractmethod
    def _start_new_process(self):
        """Start a new process
        """

    @abstractmethod
    def _stop_running_process(self, process_components):
        """Stop a running process
        """

    @abstractmethod
    def marshall_request(self, sid, request):
        """Send the given request to the model (and thus process) with the given sid"""

    def reserve_process(self, sid: str):
        if len(self._idle_processes) == 0:
            # create a new process ready to process requests
            process_components = self._start_new_process()
            process_components[0].start()
            self._idle_processes.append(process_components)

        # select the first available process and assign it to process requests for model sid
        selected_process_components = self._idle_processes[0]
        self._idle_processes = self._idle_processes[1:]
        self._sid_to_active_process_mapping[sid] = selected_process_components

    def release_process(self, sid: str):
        self._idle_processes.append(
            self._sid_to_active_process_mapping.pop(sid)
        )
        print(self._sid_to_active_process_mapping)

    def shutdown(self):
        """Stop all running processes in a clean way
        """
        assert len(self._sid_to_active_process_mapping) == 0
        for proc_data in self._idle_processes:
            self._stop_running_process(proc_data)
            proc_data[0].join()
        self._idle_processes.clear()

    def kill_all(self):
        """Send a sigkill signal to all running processes -- mostly to be used in emergencies
        """
        for worker, *_ in self._idle_processes:
            if worker.is_alive():
                worker.terminate()
        for worker, *_ in self._sid_to_active_process_mapping.values():
            if worker.is_alive():
                worker.terminate()
        self._idle_processes.clear()
        self._sid_to_active_process_mapping.clear()

    def get_active_sids(self):
        return list(self._sid_to_active_process_mapping.keys())


class ModelProcessPoolWPipe(ModelProcessPool):
    def _start_new_process(self):
        parent_conn, child_conn = mp.Pipe()
        worker = ModelWorker(
            model=self._model,
            connection=child_conn,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return (worker, parent_conn)

    def _stop_running_process(self, process_components):
        request = (TaskType.SHUTDOWN, None)
        process_components[1].send(request)

    def marshall_request(self, sid, request):
        _, connection = self._sid_to_active_process_mapping[sid]
        connection.send(request)
        return connection.recv()  # blocking


class ModelProcessPoolWQueue(ModelProcessPool):
    def _start_new_process(self):
        request_queue = mp.Queue()
        response_queue = mp.Queue()
        worker = ModelWorker(
            model=self._model,
            request_queue=request_queue,
            response_queue=response_queue,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        return (worker, request_queue, response_queue)

    def _stop_running_process(self, process_components):
        request = (TaskType.SHUTDOWN, None)
        process_components[1].put(request)

    def marshall_request(self, sid, request):
        _, request_queue, response_queue = self._sid_to_active_process_mapping[sid]
        request_queue.put(request)
        return response_queue.get()


class ModelProcessPoolWPipeShareMemory(ModelProcessPool):
    def __init__(self, model, tensor_metadata):
        super().__init__(model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._shared_tensors = {
            name: torch.empty(tensor_shape, dtype=tensor_dtype, device=device).share_memory_()
            for name, tensor_shape, tensor_dtype in tensor_metadata
        }

    def _start_new_process(self):
        parent_conn, child_conn = mp.Pipe()
        worker = ModelWorker(
            model=self._model,
            connection=child_conn,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            shared_tensors=self._shared_tensors,
        )
        return (worker, parent_conn)

    def _stop_running_process(self, process_components):
        process_components[1].send((TaskType.SHUTDOWN, None))

    def marshall_request(self, sid, request):
        _, connection = self._sid_to_active_process_mapping[sid]

        # Example: if train or evaluation request, we treat the request in a separate way
        # to avoid multiple data copies
        if request[0] == TaskType.TRAIN:
            control, payload = request
            # Copy input data into shared tensors
            self._shared_tensors["embeddings"].copy_(payload[0])
            self._shared_tensors["labels"].copy_(payload[1])
            num_offloaded_layers = payload[2]

            connection.send((control, num_offloaded_layers))
            loss = connection.recv()

            grad = self._shared_tensors["gradient"].cpu()
            return grad, loss

        else:
            connection.send(request)
            return connection.recv()
