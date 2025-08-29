from typing import Dict
from logging import INFO

import torch

from flwr.common import Config, Scalar, log

from slbd.common.constants import CLIENT_DATASET_SIZE

from slbd.algos.base_numpy_client import BaseNumpyClientAdapter
from slbd.common.profiling import measure_per_layer_time


class ARESNumpyClientAdapter(BaseNumpyClientAdapter):

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        # locally profile the model forward and backward time for every layer in the model
        log(INFO, "Profiling model on device...")
        optimizer = torch.optim.SGD(self.whole_model.parameters(), momentum=0.9)
        forward_times, backward_times, _ = measure_per_layer_time(
            model=self.whole_model,
            optimizer=optimizer,
            input_shape=(64, 3, 32, 32),
            iterations=config["iterations"],
            dry_runs=config["dry_runs"]
        )
        del optimizer

        out = {
            f"f+b{idx}": forward_times[idx] + backward_times[idx]
            for idx in range(len(forward_times))
        }
        out[CLIENT_DATASET_SIZE] = self.torch_client.get_trainloader_size()
        log(INFO, "Profiling complete %s", str(out))
        return out
