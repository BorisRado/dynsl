import torch

from slbd.client.abstract_torch_client import AbstractTorchClient

from exp.src.common.stochasticity import set_seed
from exp.src.common.background_process_manager import BackgroundProcessManager
from exp.src.data.loading import load_client_trainset
from exp.src.data.utils import get_dataloader


class MOTorchClient(AbstractTorchClient):

    def __init__(
        self,
        cid,
        min_layers,
        batch_size,
        dataset_config,
        partitioning_config,
        seed,
        bg_proc_delay=0.,
        bg_proc_duration=0.,
        bg_num_procs=0,
        bg_device="cpu"
    ):
        self.cid = int(cid)
        self.min_layers = min_layers
        self.batch_size = batch_size
        self.seed = seed
        self.trainset = load_client_trainset(
            client_idx=self.cid,
            dataset_config=dataset_config,
            partitioning_config=partitioning_config,
            seed=seed
        )
        self.bg_proc_delay = bg_proc_delay
        self.bg_proc_duration = bg_proc_duration
        self.bg_num_procs = bg_num_procs
        assert bg_device in {"cpu", "cuda"}
        bg_device = torch.device("cuda" if torch.cuda.is_available() and bg_device == "cuda" else "cpu")
        print(bg_num_procs, bg_proc_delay, bg_proc_duration, bg_device)
        self.bg_proc_mng = BackgroundProcessManager(
            num_processes=self.bg_num_procs, bg_device=bg_device
        )

    def fit(self, config, model_trainer):
        trainloader = get_dataloader(self.trainset, batch_size=self.batch_size)

        self.bg_proc_mng.resume_processes(
            delay=self.bg_proc_delay,
            duration=self.bg_proc_duration,
        )
        set_seed(self.seed + config["epoch_idx"])
        if self.trainset._dataset_type == "cv":
            for batch in trainloader:
                model_trainer(batch["img"], batch["label"])
        else:
            for batch in trainloader:
                model_trainer(
                    input_vars=(batch["input_ids"], batch["attention_mask"]),
                    labels=batch["labels"]
                )
        self.bg_proc_mng.pause_processes()

    def evaluate(self, config, model_trainer):
        valloader = get_dataloader(self.trainset, batch_size=self.batch_size, shuffle=False)
        for batch in valloader:
            model_trainer.evaluate(batch["img"], batch["label"])

    def get_trainloader_size(self):
        return len(self.trainset) // self.batch_size
