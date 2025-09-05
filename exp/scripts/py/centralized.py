import time

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from slbd.torch.utils import OptimizerGenerator, init_optimizer

from exp.src.data.loading import load_client_trainset
from exp.src.data.utils import get_dataloader
from exp.src.common.stochasticity import set_seed
from exp.src.common.other import set_torch_flags


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.general.seed)
    model = instantiate(cfg.model).to(device)

    dataset = load_client_trainset(0, cfg.data, cfg.partitioning, cfg.general.seed)
    trainloader = get_dataloader(dataset, batch_size=16)
    valloader = get_dataloader(dataset, batch_size=16, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer_config_generator = OptimizerGenerator(cfg.optimizer.optimizer, cfg.optimizer.scheduler)
    start_time = time.time()

    for epoch in range(1, cfg.general.rounds + 1):
        set_seed(cfg.general.seed+epoch)

        optimizer = init_optimizer(
            model.parameters(),
            **optimizer_config_generator.get_optimizer_config()
        )
        model.train()
        for batch in trainloader:
            img, label = batch["img"].to(device), batch["label"].to(device)
            logits = model(img)
            optimizer.zero_grad()
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        set_seed(10)
        with torch.no_grad():
            for batch in valloader:
                img, label = batch["img"].to(device), batch["label"].to(device)
                logits = model(img)
                loss = criterion(logits, label)
                total_loss += loss.item() * label.size(0)
                predicted = torch.argmax(logits, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)
            accuracy = correct / total
            avg_loss = total_loss / total
        optimizer_config_generator.end_epoch_callback()
        print(f"Train Accuracy: {accuracy:.6f}, Train Loss: {avg_loss:.6f}")

    print(f"Runtime {time.time() - start_time:.2f}")


if __name__ == "__main__":
    set_torch_flags()
    run()
