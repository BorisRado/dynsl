import sys
import time
import threading
import subprocess
import traceback

import torch

from slbd.client.abstract_torch_client import AbstractTorchClient

from exp.src.common.k8s import get_node_name
from exp.src.common.stochasticity import set_seed
from exp.src.common.background_process_manager import BackgroundProcessManager
from exp.src.common.gpu_control import throttle_gpu, restore_gpu
from exp.src.data.utils import get_dataloader


class MOTorchClient(AbstractTorchClient):

    def __init__(
        self,
        cid,
        min_layers,
        batch_size,
        trainset,
        num_epochs,
        seed,
        bg_proc_delay=0.,
        bg_proc_duration=0.,
        bg_num_procs=0,
        bg_device="cpu",
        reduced_network_throughput=-1,
        slowdown_gpu=False,
    ):
        self.min_layers = min_layers
        self.batch_size = batch_size
        self.seed = seed
        self.num_epochs = num_epochs
        self.trainset = trainset
        self.bg_proc_delay = bg_proc_delay
        self.bg_proc_duration = bg_proc_duration
        self.bg_num_procs = bg_num_procs
        self.reduced_network_throughput = reduced_network_throughput
        self.node_name = get_node_name()
        self.slow_down_gpu = slowdown_gpu and self.node_name == "jn1"
        assert bg_device in {"cpu", "cuda"}
        print("...", self.reduced_network_throughput)
        print("Slowing down GPU?", self.slow_down_gpu)
        if self.slow_down_gpu:
            restore_gpu()
        if self.reduced_network_throughput and self.node_name == "op1":
            subprocess.run(["apt-get", "update"])
            subprocess.run(["apt-get", "install", "-y", "iproute2"])
        bg_device = torch.device("cuda" if torch.cuda.is_available() and bg_device == "cuda" else "cpu")
        if self.bg_num_procs > 0:
            self.bg_proc_mng = BackgroundProcessManager(
                num_processes=self.bg_num_procs, bg_device=bg_device
            )

    def fit(self, config, model_trainer):
        trainloader = get_dataloader(self.trainset, batch_size=self.batch_size)

        if self.bg_num_procs > 0:
            self.bg_proc_mng.resume_processes(
                delay=self.bg_proc_delay,
                duration=self.bg_proc_duration,
            )
        if self.slow_down_gpu:
            print("Slowing down GPU", flush=True)
            throttle_gpu(5)
        if self.reduced_network_throughput > 0 and self.node_name == "op1":
            def _modify_throughput():
                try:
                    time.sleep(5)
                    print(f"Will reduce throughput to: {self.reduced_network_throughput}")
                    subprocess.run([
                        "./modify_throughput.sh", "apply",
                        str(self.reduced_network_throughput),
                        str(self.reduced_network_throughput)
                    ], check=True)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
            threading.Thread(target=_modify_throughput, daemon=True).start()
        set_seed(self.seed + config["epoch_idx"])
        for epoch_idx in range(self.num_epochs):
            print("Running epoch", epoch_idx)
            if self.trainset._dataset_type == "cv":
                n = 0
                for batch in trainloader:
                    model_trainer(batch["img"], batch["label"])
                    n += 1
                print("Have completed the entire trainloader!", n, len(trainloader))
            else:
                for batch in trainloader:
                    model_trainer(
                        input_vars=(batch["input_ids"], batch["attention_mask"]),
                        labels=batch["labels"]
                    )
        if self.bg_num_procs > 0:
            self.bg_proc_mng.pause_processes()
        if self.slow_down_gpu:
            print("Restoring GPU", flush=True)
            restore_gpu()

        if self.reduced_network_throughput > 0 and self.node_name == "op1":
            try:
                subprocess.run(["./modify_throughput.sh", "clear"], check=True)
            except:
                traceback.print_exc(file=sys.stderr)

    def evaluate(self, config, model_trainer):
        valloader = get_dataloader(self.trainset, batch_size=self.batch_size, shuffle=False)
        for batch in valloader:
            model_trainer.evaluate(batch["img"], batch["label"])

    def get_trainloader_size(self):
        # print("Per-epoch batches", len(self.trainset) // self.batch_size)
        # print("Tot # batches:", self.num_epochs * (len(self.trainset) // self.batch_size))
        return self.num_epochs * (len(self.trainset) // self.batch_size)
