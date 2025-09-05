from slbd.algos.base_strategy import BaseStrategy


class SplitFedStrategy(BaseStrategy):
    def __init__(self, version, **kwargs):
        super().__init__(**kwargs)
        assert version in {"v1", "v2"}
        self._common_server_model = version == "v2"

    def configure_fit(self, server_round, model_state_dict, client_manager, client_layer_allocator):

        config = self.optimizer_config_generator.get_optimizer_config()
        if self.config_client_fit_fn:
            config |= self.config_client_fit_fn(server_round)

        clients = self.sample_epoch_clients(client_manager, True)
        num_initial_client_layers = client_layer_allocator \
            .initialize_epoch(clients, client_manager, True)

        all_clients_fit_ins = self._setup_epoch_clients(
            server_round, clients, num_initial_client_layers, model_state_dict, True, config
        )
        if self._common_server_model:
            all_server_models_fit_ins = self._setup_epoch_server_models(
                [("", min(num_initial_client_layers), [proxy.cid for proxy in clients])],
                model_state_dict, True, config
            )
        else:
            all_server_models_fit_ins = self._setup_epoch_server_models(
                [(proxy.cid, min(num_initial_client_layers), [proxy.cid]) for proxy in clients],
                model_state_dict, True, config
            )
        return all_clients_fit_ins, all_server_models_fit_ins

    def configure_evaluate(self, server_round, model_state_dict, client_manager, client_layer_allocator):
        config = {}

        clients = self.sample_epoch_clients(client_manager, False)
        num_initial_client_layers = client_layer_allocator \
            .initialize_epoch(clients, client_manager, False)

        all_clients_fit_ins = self._setup_epoch_clients(
            server_round, clients, num_initial_client_layers, model_state_dict, False, config
        )
        if self._common_server_model:
            all_server_models_fit_ins = self._setup_epoch_server_models(
                [("", min(num_initial_client_layers), [proxy.cid for proxy in clients])],
                model_state_dict, True, config
            )
        else:
            all_server_models_fit_ins = self._setup_epoch_server_models(
                [(proxy.cid, min(num_initial_client_layers), [proxy.cid]) for proxy in clients],
                model_state_dict, False, config
            )
        return all_clients_fit_ins, all_server_models_fit_ins

    def route_client_request_to_server_model(self, client_id):
        if self._common_server_model:
            return ""
        return client_id
