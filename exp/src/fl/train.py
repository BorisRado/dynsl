from copy import deepcopy
from typing import Dict, Tuple
from collections import OrderedDict

import ray
import wandb
import torch
from flwr.common import NDArrays, Scalar, parameters_to_ndarrays
from flwr.client import NumPyClient as FlwrNumpyClient
from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

from slbd.torch.utils import init_optimizer
from slbd.torch.utils import OptimizerGenerator

from exp.src.common.stochasticity import set_seed
from exp.src.data.utils import get_dataloader
from exp.src.common.stochasticity import set_seed
from exp.src.model.utils import test_model


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def train_flower(
    model,
    client_init_kwargs,
    trainsets,
    strategy,
    n_rounds,
    seed,
    client_resources={"num_cpus": 2, "num_gpus": 0.2},
):

    if not ray.is_initialized():
        ray.init()

    trainsets_refs = [ray.put(ts) for ts in trainsets]

    def client_fn(cid: str):
        """Create and return an instance of Flower `Client`."""
        cid = int(cid)
        ts = ray.get(trainsets_refs[cid])

        return FlowerClient(
            trainset=ts,
            model=deepcopy(model),
            seed=seed,
            **client_init_kwargs
        ).to_client()

    cr = client_resources.copy()
    if not torch.cuda.is_available():
        _ = cr.pop("num_gpus", None)

    set_seed(seed)
    history = start_simulation(
        client_fn=client_fn,
        num_clients=len(trainsets),
        config=ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources=cr,
        keep_initialised=True,
    )
    print(history)

    # history.losses_distributed
    return history


def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    out = {}
    ns = [num_examples for num_examples, _ in metrics]
    for k in metrics[0][1].keys():
        vals = [num_examples * m[k] for num_examples, m in metrics]
        out[k] = sum(vals) / sum(ns)
    return out


class CustomFedAvg(FedAvg):
    def __init__(self, model, optimizer_config, testloader, **kwargs):
        self.model = model
        delete_kwargs = ["config_client_fit_fn", "num_server_models"]
        for k in delete_kwargs:
            if k in kwargs:
                del kwargs[k]
        self.optimizer_config_generator = OptimizerGenerator(
            optimizer_cfg=optimizer_config["optimizer"],
            scheduler_cfg=optimizer_config["scheduler"],
        )
        self.testloader = testloader
        super().__init__(
            **kwargs,
            on_fit_config_fn=lambda server_round: self.optimizer_config_generator.get_optimizer_config() | {"epoch_idx": server_round},
            fit_metrics_aggregation_fn=weighted_average
        )

    def aggregate_fit(self, server_round, results, failures):
        self.optimizer_config_generator.end_epoch_callback()
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if wandb.run is not None:
            wandb.log(metrics)
        return parameters, metrics

    def evaluate(self, server_round, parameters):
        set_parameters(self.model, parameters_to_ndarrays(parameters))
        test_loss, test_acc = test_model(self.model, self.testloader)
        print("evaluated model on test set", test_acc, test_loss)
        if wandb.run is not None:
            wandb.log({
                "test/accuracy": test_acc,
                "test/loss": test_loss,
            })


class FlowerClient(FlwrNumpyClient):
    """Adapter class for Flower clients using NumPy in split learning context."""

    def __init__(
        self,
        *,
        cid,
        trainset,
        model,
        batch_size,
        num_epochs,
        seed,
        min_layers,
    ):
        self.cid = cid
        self.model = model
        self.trainset = trainset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        optimizer_config = {k: v for k, v in config.items() if k.startswith("optim.")}
        set_seed(self.seed+config["epoch_idx"])
        set_parameters(self.model, parameters)

        self.model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()

        optimizer = init_optimizer(self.model.parameters(), **optimizer_config)
        trainloader = get_dataloader(self.trainset, self.batch_size, True)

        tot_loss = 0.0
        total_samples = 0
        for _ in range(self.num_epochs):
            for batch in trainloader:
                optimizer.zero_grad()
                logits = self.model(batch["img"].to(self.device))
                loss = criterion(logits, batch["label"].to(self.device))
                batch_size = batch["img"].shape[0]
                tot_loss += loss.item() * batch_size
                total_samples += batch_size
                loss.backward()
                optimizer.step()
        avg_loss = tot_loss / total_samples
        return self.get_parameters({}), len(trainloader.dataset), {"train/loss": avg_loss}

    def get_parameters(self, config):
        return get_parameters(self.model)
