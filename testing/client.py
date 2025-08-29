from slbd.client.app import start_client
from slbd.client.abstract_torch_client import AbstractTorchClient
from testing.model import Model
from testing.utils import get_train_dataloader, set_seed

class Client(AbstractTorchClient):

    def __init__(self, cid):
        _ = (cid, )
        self.trainloader = get_train_dataloader()

    def fit(self, config, model_trainer):
        _ = (config, )
        # Define transformations for the data
        set_seed()

        for img, label in self.trainloader:
            model_trainer(img, label)

    def evaluate(self, config, model_trainer):
        _ = (config,)
        for img, label in self.trainloader:
            model_trainer.evaluate(img, label)

    def get_trainloader_size(self):
        return len(self.trainloader)


if __name__ == "__main__":
    set_seed()

    start_client(
        config_file="testing/client_config.yaml",
        client_fn=Client,
        model=Model(),
    )
