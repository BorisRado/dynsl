from flwr.common import ndarrays_to_parameters, bytes_to_ndarray

from slbd.algos.base_strategy import BaseStrategy
from slbd.torch.utils import create_classification_head
from slbd.torch.model_utils import extract_submodel


class SFSLStrategy(BaseStrategy):
    def __init__(self, clf_head_dimensions, num_classes, num_client_layers, **kwargs):
        super().__init__(**kwargs)
        self.clf_head_dimensions = clf_head_dimensions

        # get the initial state dict of the classification head
        client_encoder = extract_submodel(self.model, 0, num_client_layers)
        clf_head = create_classification_head(
            encoder_model=client_encoder,
            num_classes=num_classes,
            fc_dims=self.clf_head_dimensions
        )
        self.clf_head_parameters = [p.detach().numpy() for p in clf_head.state_dict().values()]

    def configure_fit(self, *args, **kwargs):
        all_clients_fit_ins, all_server_models_fit_ins = super().configure_fit(*args, **kwargs)
        # append the parameters of the client classification head to the instructions
        clf_head_parameters = ndarrays_to_parameters(self.clf_head_parameters)
        for _, fit_ins in all_clients_fit_ins:
            fit_ins.parameters.tensors += clf_head_parameters.tensors
        return all_clients_fit_ins, all_server_models_fit_ins

    def aggregate_fit(self, server_round, client_results, server_results, layer_allocator, failures):
        nhp = len(self.clf_head_parameters)
        client_clf_head_parameters = [[] for _ in range(nhp)]
        for _, fit_res in client_results:
            fit_res.parameters.tensors, clf_head_params = \
                fit_res.parameters.tensors[:-nhp], fit_res.parameters.tensors[-nhp:]
            clf_head_params = [bytes_to_ndarray(tensor) for tensor in clf_head_params]

            processed_dp = int(fit_res.metrics["_proc0"])
            for idx, param in enumerate(clf_head_params):
                client_clf_head_parameters[idx].append((param, processed_dp))
        self.clf_head_parameters = [self._aggregate_layer_parameters(vals) for vals in client_clf_head_parameters]
        return super().aggregate_fit(server_round, client_results, server_results, layer_allocator, failures)
