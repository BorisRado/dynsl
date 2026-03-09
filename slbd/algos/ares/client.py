from typing import Dict
from logging import INFO

import torch

from flwr.common import Config, Scalar, log

from slbd.algos.base_numpy_client import BaseNumpyClientAdapter
from slbd.common.profiling import profile_cumulative_per_layer_time_individual_layers


class ARESNumpyClientAdapter(BaseNumpyClientAdapter):

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        # locally profile the model forward and backward time for every layer in the model
        log(INFO, "Profiling model on device...")
        iterations = config.pop("iterations")
        dry_runs = config.pop("dry_runs")

        per_layer_times = profile_cumulative_per_layer_time_individual_layers(
            model=self.whole_model,
            optimizer_name="adam",
            batch_size=self.torch_client.batch_size,
            iterations=iterations,
            dry_runs=dry_runs,
        )

        out = {f"f+b{idx}": per_layer_times[idx] for idx in range(len(per_layer_times))}
        log(INFO, "Profiling complete %s", str(out))
        torch.cuda.empty_cache()
        return out
