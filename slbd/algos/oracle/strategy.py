from slbd.algos.base_strategy import BaseStrategy


class OracleStrategy(BaseStrategy):
    def configure_fit(self, server_round, model_state_dict, client_manager, client_layer_allocator):
        if self.num_server_models in {1, -1}:
            return super() \
                .configure_fit(server_round, model_state_dict, client_manager, client_layer_allocator)

        print("Using separate model for every split point")
        client_manager.wait_for(self.min_available_clients)
        config = self.optimizer_config_generator.get_optimizer_config()
        if self.config_client_fit_fn:
            if isinstance(self.config_client_fit_fn, dict):
                config |= self.config_client_fit_fn
            else:
                config |= self.config_client_fit_fn(server_round)

        clients = self.sample_epoch_clients(client_manager, True)
        if len(clients) == 0:
            return [], []
        num_initial_client_layers = client_layer_allocator \
            .initialize_epoch(clients, client_manager, True)

        all_clients_fit_ins = self._setup_epoch_clients(
            server_round, clients, num_initial_client_layers, model_state_dict, True, config
        )
        server_configs = []
        for split_point in set(num_initial_client_layers):
            clients_w_sp = [
                client.cid
                for ncl, client in zip(num_initial_client_layers, clients)
                if ncl == split_point
            ]
            print(f"SP {split_point} -- {len(clients_w_sp)}")
            server_configs.append((
                str(split_point), split_point, clients_w_sp
            ))

        all_server_models_fit_ins = self._setup_epoch_server_models(
            server_configs,
            model_state_dict, True, config
        )
        return all_clients_fit_ins, all_server_models_fit_ins
