import time

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from slbd.torch.utils import OptimizerGenerator, init_optimizer

from exp.src.data.loading import load_client_trainset
from exp.src.data.utils import get_dataloader
from exp.src.common.stochasticity import set_seed
from exp.src.common.other import set_torch_flags


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if "img" in batch:
                img = batch["img"].to(device)
                logits = model(img)
                label = batch["label"].to(device, non_blocking=True)
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                label = batch["labels"].to(device, non_blocking=True)

            loss = torch.nn.CrossEntropyLoss()(logits, label)
            total_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    print(f"Accuracy {accuracy:.3f}")
    print(f"Avg loss {avg_loss:.3f}")


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg.general.seed)
    model = instantiate(cfg.model)
    model.to(device)

    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    print("Device: ", device)

    dataset = load_client_trainset(0, cfg.data, cfg.partitioning, cfg.general.seed)
    print("Centralized dataset size", len(dataset))
    trainloader = get_dataloader(dataset, batch_size=cfg.client.init_kwargs.batch_size)
    criterion = torch.nn.CrossEntropyLoss()

    evaluate_model(model, trainloader, device)

    optimizer_config_generator = OptimizerGenerator(cfg.optimizer.optimizer, cfg.optimizer.scheduler)
    start_time = time.time()

    for epoch in range(1, cfg.general.rounds + 1):
        set_seed(cfg.general.seed+epoch)

        optimizer = init_optimizer(
            model.parameters(),
            **optimizer_config_generator.get_optimizer_config()
        )
        model.train()
        tot_loss = 0.
        for batch in trainloader:
            if "img" in batch:
                img = batch["img"].to(device)
                logits = model(img)
                label = batch["label"].to(device, non_blocking=True)
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids, attention_mask)
                label = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = criterion(logits, label)
            # print(logits.min().min().item(), logits.max().max().item(), loss.item())
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        print(f"Loss {tot_loss / len(trainloader)}")

    evaluate_model(model, trainloader, device)
    print(f"Runtime {time.time() - start_time:.2f}")


if __name__ == "__main__":
    set_torch_flags()
    run()
