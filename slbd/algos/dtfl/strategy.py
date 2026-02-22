from flwr.common import ndarrays_to_parameters, bytes_to_ndarray

from slbd.algos.base_strategy import BaseStrategy
from slbd.torch.utils import create_classification_head
from slbd.torch.model_utils import extract_submodel
from slbd.common.constants import CLIENT_NUM_RECEIVED_LAYERS


class DTFLStrategy(BaseStrategy):
    def __init__(self, clf_head_dimensions, num_classes, all_split_points, **kwargs):
        super().__init__(**kwargs)
        self.clf_head_dimensions = clf_head_dimensions

        # get the initial state dict of the classification head
        self.all_split_points = all_split_points
        self.all_clf_head_parameters = {}
        self._client_epoch_layers = {}
        for split_point in all_split_points:
            client_encoder = extract_submodel(self.model, 0, split_point)
            clf_head = create_classification_head(
                encoder_model=client_encoder,
                num_classes=num_classes,
                fc_dims=self.clf_head_dimensions
            )
            self.all_clf_head_parameters[split_point] = [p.detach().numpy() for p in clf_head.state_dict().values()]

    def configure_fit(self, *args, **kwargs):
        all_clients_fit_ins, all_server_models_fit_ins = super().configure_fit(*args, **kwargs)
        # append the parameters of the client classification head to the instructions
        for client_proxy, fit_ins in all_clients_fit_ins:
            # get how many layers the client is training
            num_client_layers = fit_ins.config[CLIENT_NUM_RECEIVED_LAYERS]
            self._client_epoch_layers[client_proxy.cid] = num_client_layers

            # attach the corresponding weights to the client parameters
            clf_head_parameters = ndarrays_to_parameters(self.all_clf_head_parameters[num_client_layers])
            fit_ins.parameters.tensors += clf_head_parameters.tensors

        return all_clients_fit_ins, all_server_models_fit_ins

    def aggregate_fit(self, server_round, client_results, server_results, layer_allocator, failures):
        for split_point in self.all_clf_head_parameters:
            print(f"Aggregating clf head for split point {split_point}")
            nhp = len(self.all_clf_head_parameters[split_point])
            print(nhp)
            client_clf_head_parameters = [[] for _ in range(nhp)]

            for client_proxy, fit_res in client_results:
                if self._client_epoch_layers[client_proxy.cid] != split_point:
                    continue
                fit_res.parameters.tensors, clf_head_params = \
                    fit_res.parameters.tensors[:-nhp], fit_res.parameters.tensors[-nhp:]
                clf_head_params = [bytes_to_ndarray(tensor) for tensor in clf_head_params]

                processed_dp = int(fit_res.metrics["_proc0"])
                for idx, param in enumerate(clf_head_params):
                    client_clf_head_parameters[idx].append((param, processed_dp))

            nr = len(client_clf_head_parameters[0])
            print(f"Received {nr} results for split point...")
            if nr > 0:
                self.all_clf_head_parameters[split_point] = \
                    [self._aggregate_layer_parameters(vals) for vals in client_clf_head_parameters]
        self._client_epoch_layers.clear()
        return super().aggregate_fit(server_round, client_results, server_results, layer_allocator, failures)
