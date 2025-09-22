from tqdm import tqdm

from dotenv import load_dotenv
import wandb
import os
import json


def run(include_running):
    or_filters = [{"state": "finished"}]
    if include_running:
        or_filters.append({"state": "running"})

    api = wandb.Api(timeout=30)
    runs = api.runs(per_page=100000, filters={"$or": or_filters})
    configs = [(run.id, run.config) for run in runs]
    os.makedirs(".wandb_cache", exist_ok=True)

    for run_id, config in tqdm(configs):
        with open(f".wandb_cache/{run_id}.json", "w") as f:
            json.dump(config, f)


if __name__ == "__main__":
    load_dotenv()

    include_running = input("Include running? [y/n] ").lower()
    assert include_running in {"y", "n"}
    include_running = include_running == "y"
    print(f"Including running? {include_running}")
    run(include_running)
