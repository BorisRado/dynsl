import copy
import timeit
from collections import OrderedDict
from logging import INFO
from typing import Dict, Optional, Tuple

from flwr.common import (
    Parameters,
    Scalar,
    log,
)
from flwr.server.client_manager import ClientManager
from flwr.server.history import History
from flwr.server.server import (
    fit_clients,
    evaluate_clients,
    FitResultsAndFailures,
    EvaluateResultsAndFailures,
    Server as FlwrServer
)

from slbd.server.strategy import Strategy
from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.server.server_model.manager.manager import ServerModelManager
from slbd.common.constants import CLIENT_ID_CONFIG_KEY


class Server(FlwrServer):
    """Flower server."""

    def __init__(
        self,
        *,
        server_model_manager: ServerModelManager,
        client_manager: ClientManager,
        client_layer_allocator: ClientLayerAllocator,
        strategy: Strategy,
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.server_model_manager: ServerModelManager = server_model_manager
        self.server_model_manager.set_strategy(strategy)
        self.client_layer_allocator = client_layer_allocator

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters()

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(
                server_round=current_round,
                timeout=timeout,
            )
            if res_fit is not None:
                parameters_prime, fit_metrics, _ = res_fit
                self.parameters.update({k: v for k, v in parameters_prime.items() if v is not None})
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )

            # Evaluate model using strategy implementation
            # res_cen = self.strategy.evaluate(
            #     current_round,
            #     parameters=self.parameters,
            # )
            # if res_cen is not None:
            #     loss_cen, metrics_cen = res_cen
            #     log(
            #         INFO,
            #         "fit progress: (%s, %s, %s, %s)",
            #         current_round,
            #         loss_cen,
            #         metrics_cen,
            #         timeit.default_timer() - start_time,
            #     )
            #     history.add_loss_centralized(server_round=current_round, loss=loss_cen)
            #     history.add_metrics_centralized(
            #         server_round=current_round, metrics=metrics_cen
            #     )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed is not None:
                loss_fed, evaluate_metrics_fed, _ = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        self.server_model_manager.shutdown()
        self.server_model_manager.clean_resources()
        log(INFO, "FL finished in %s", elapsed)
        return history, elapsed

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions, server_model_configs = self.strategy.configure_evaluate(
            server_round=server_round,
            model_state_dict=self.parameters,
            client_manager=self._client_manager,
            client_layer_allocator=self.client_layer_allocator,
        )

        # make a copy of instructions to avoid modifying the original ones in case multiple
        # ins have the same config dictionary
        client_instructions = [(proxy, copy.deepcopy(ins)) for proxy, ins in client_instructions]
        for proxy, ins in client_instructions:
            ins.config[CLIENT_ID_CONFIG_KEY] = proxy.cid

        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            INFO,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        self.server_model_manager.initialize_server_models(server_model_configs, is_training=False)

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        self.server_model_manager.end_round()
        self.client_layer_allocator.finalize_epoch()
        return loss_aggregated, metrics_aggregated, (results, failures)

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions, server_model_configs = self.strategy.configure_fit(
            server_round=server_round,
            model_state_dict=self.parameters,
            client_manager=self._client_manager,
            client_layer_allocator=self.client_layer_allocator,
        )

        client_instructions = [(proxy, copy.deepcopy(ins)) for proxy, ins in client_instructions]
        for proxy, ins in client_instructions:
            ins.config[CLIENT_ID_CONFIG_KEY] = proxy.cid

        if not client_instructions:
            log(INFO, "fit_round %s: no clients selected, cancel", server_round)
            return None
        log(
            INFO,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # start the server proxies
        self.server_model_manager.initialize_server_models(server_model_configs, is_training=True)

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "fit_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )
        server_fit_res = self.server_model_manager.collect_server_fit_results()
        self.server_model_manager.end_round()

        # Aggregate training results
        aggregated_result: Tuple[
            OrderedDict,
            Dict[str, Scalar],
        ] = self.strategy.aggregate_fit(
            server_round=server_round,
            client_results=results,
            server_results=server_fit_res,
            layer_allocator=self.client_layer_allocator,
            failures=failures
        )

        parameters_aggregated, metrics_aggregated = aggregated_result
        self.client_layer_allocator.finalize_epoch()
        return (
            parameters_aggregated,
            metrics_aggregated,
            (results, failures)
        )

    def _get_initial_parameters(self) -> OrderedDict:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[OrderedDict] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial global parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters the server model manager")
        parameters: OrderedDict = self.server_model_manager.get_global_model_params()
        return parameters

    def get_client_layer_allocator(self):
        return self.client_layer_allocator
