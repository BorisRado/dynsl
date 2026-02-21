import copy

import torch

from slbd.server.server_model.manager import ServerModelManager


class BaseServerModelManager(ServerModelManager):

    def __init__(
        self,
        model,
    ) -> None:
        super().__init__()
        self.base_model = model

    def _concat_data(self, embeddings, labels):
        sizes = [b.shape[0] for b in labels]
        if isinstance(embeddings[0], torch.Tensor):
            emb_shape = embeddings[0].shape[1:]
            combined_embeddings = torch.empty(
                (sum(sizes), *emb_shape),
                dtype=embeddings[0].dtype,
                pin_memory=True
            )
            torch.cat(embeddings, dim=0, out=combined_embeddings)
            embeddings = combined_embeddings
        else:
            embeddings = tuple(
                torch.cat([b[i] for b in embeddings])
                for i in range(len(embeddings[0]))
            )

        labels = torch.cat(labels, dim=0)
        return embeddings, labels, (sizes,)

    def _split_data(self, loss, gradient, control):
        assert len(control) == 1
        sizes = control[0]
        return [(g.numpy(), loss) for g in gradient.split(sizes, dim=0)]

    def get_global_model_params(self):
        return copy.deepcopy(self.base_model.state_dict())

    async def marshall_weight_integration_request(self, model_id, client_parameters, cid=None, client_optimizer_states=None):
        pass
