import time
import random
import itertools
from typing import List, Dict, Optional, Tuple

from slbd.algos.dynsl.layer_allocator import DynSLClientLayerAllocator


class DynSLClientLayerAllocatorV2(DynSLClientLayerAllocator):

    def __init__(
        self,
        max_client_layers: int,
        num_model_layers: int,
        reallocation_frequency: float,
        gamma: float,
        num_concat_batches: int,
        layer_allocation_optimizer: str,
        meaningful_split_points: List[int],
        cache_client_configuration: bool,
        modeled_layer_complexities: List[float]
    ):
        """
        Initialize the LayerAllocator.

        Args:
            max_client_layers: Maximum number of layers a client can handle
            num_model_layers: Total number of model layers
            alpha: smoothing factor for layer adjustment
        """
        self._gamma = gamma
        self._per_client_min_layers: Dict[str, int] = {}
        self._per_client_max_layers: Dict[str, int] = {}
        self._num_model_layers: int = num_model_layers
        assert layer_allocation_optimizer in {"greedy", "genetic"}
        self.layer_allocation_optimizer = layer_allocation_optimizer

        # per-client dataset and pending batches
        self._dataset_sizes: Dict[str, int] = {}
        self._pending_batches: Dict[str, int] = {}

        self._allocated_layer_history = []

        # rolling averages
        self._client_training_times: Dict[str, float] = {}
        self._client_round_times: Dict[str, float] = {}

        # we will call the reallocation function with the following frequency
        self.reallocation_frequency = reallocation_frequency
        self._reallocation_stop_event = None
        self._num_concat_batches = num_concat_batches
        self.meaningful_split_points = [msp for msp in meaningful_split_points if msp <= max_client_layers]
        self._cid_to_device_type_mapping = {}

        self.cache_client_configuration = cache_client_configuration
        self._cached_num_layers = {}
        print("Caching?", self.cache_client_configuration)
        self._max_client_layers: int = self.get_previous_split_point(max_client_layers)

        # self._baseline_computational_model_per_split_point is a dictionary that, for every split
        # point (key), contains the computational requirements of training the model up to that
        # point.
        print(modeled_layer_complexities)
        modeled_layer_complexities = [modeled_layer_complexities[idx] for idx in range(len(modeled_layer_complexities))]
        print(modeled_layer_complexities)
        comput_compl_per_sp = {
            start_idx: sum(modeled_layer_complexities[:start_idx])
            for start_idx in self.meaningful_split_points
        }
        self.complexity_factors = {
            (a, b): comput_compl_per_sp[a] / comput_compl_per_sp[b]
            for a, b in itertools.product(meaningful_split_points, meaningful_split_points)
        }

        self._server_training_times = {
            start_idx: sum(modeled_layer_complexities[start_idx:])
            for start_idx in self.meaningful_split_points
        }
        print("Meaningful split points: ", self.meaningful_split_points)
        print("Static server training times:", self._server_training_times)
        self.known_client_training_times_per_layer = {}

    def trigger_reallocation(self) -> None:
        start_time = time.time()
        # print(self._client_train_batches_count)
        print({k: v for k, v in self._client_allocated_layers.items() if k in self._client_layer_diff})
        if any(v < 6 for v in self._client_train_batches_count.values()):
            print("Not enough data to try reallocating...")
            return
        if len(self._client_layer_diff) == 0:
            print("All have completed training. Exiting...")
            return
        if self.layer_allocation_optimizer == "genetic":
            best_config = self._genetic_reallocation()
        elif self.layer_allocation_optimizer == "greedy":
            best_config = self._greedy_reallocation()

        J_current = self._compute_objective(self._client_allocated_layers)
        J_new = self._compute_objective(best_config)
        delta = J_current - J_new
        print(f"Current cost: {J_current:.3f}")
        print(f"Found cost: {J_new:.3f}")

        if delta > self._gamma:
            print(f"[GA] Adopt new configuration (ΔJ={delta:.3f} > γ={self._gamma})")
            for cid, new_layers in best_config.items():
                self._client_layer_diff[cid] = new_layers - self._client_allocated_layers[cid]
                self._client_train_batches_count[cid] = 0
        else:
            print(f"[GA] Keep current configuration (ΔJ={delta:.3f} ≤ γ={self._gamma})")
        print(f"Total reallocation time: {time.time() - start_time:4f}")

    def _genetic_reallocation(self, num_generations=200, pop_size=50, mutation_rate=0.2):
        clients = list(self._epoch_clients)
        split_points = self.meaningful_split_points
        baseline = [self._client_allocated_layers[cid] for cid in clients]

        def fitness(L):
            layer_config = {cid: L[i] for i, cid in enumerate(clients)}
            total_time = self._compute_objective(layer_config)
            return -total_time

        def local_mutation(L):
            """Mutate one or more clients by ±1 split point."""
            new_L = L.copy()
            for i in range(len(L)):
                if random.random() < mutation_rate:
                    pos = split_points.index(L[i])
                    step = random.choice([-1, +1])
                    new_pos = max(
                        0,
                        min(
                            len(split_points) - 1,
                            pos + step
                        )
                    )
                    new_L[i] = split_points[new_pos]
            return new_L

        # --- Initialize population near baseline ---
        population = [baseline.copy()]
        while len(population) < pop_size:
            child = local_mutation(baseline)
            population.append(child)

        # --- Evolution loop ---
        best_solution = baseline.copy()
        best_fitness = fitness(best_solution)
        iter_wo_improvement = 0

        for gen in range(num_generations):
            scores = [fitness(ind) for ind in population]
            ranked = sorted(zip(scores, population), reverse=True)
            best_solution = ranked[0][1]
            tmp_best_fitness = ranked[0][0]
            # print(tmp_best_fitness, best_fitness)
            if tmp_best_fitness > best_fitness:
                iter_wo_improvement = 0
            iter_wo_improvement += 1
            best_fitness = ranked[0][0]
            if iter_wo_improvement > 10:
                print("Breaking", gen)
                break

            # print(f"[GA] Gen {gen}: best total_time = {-best_fitness:.3f}")
            next_pop = [ind for _, ind in ranked[:4]]
            # Generate mutants
            while len(next_pop) < pop_size:
                parent = random.choice(ranked[:10])[1]
                child = local_mutation(parent)
                next_pop.append(child)

            population = next_pop

        return {cid: best_solution[i] for i, cid in enumerate(clients)}

    def _compute_objective(self, layer_config):
        pred_client_times = {}
        server_time = 0.
        for cid, L in layer_config.items():
            B = self._pending_batches[cid]
            server_time += B * self._server_training_times[L] / self._num_concat_batches
            pred_client_times[cid] = self.compute_client_time(cid, L)

        # additionally penalize every layer change
        reg_term = sum([
            layer_config[cid] != self._client_allocated_layers[cid] for cid in layer_config
        ])
        max_client_time = max(pred_client_times.values())
        total_time = max(server_time, max_client_time) + reg_term / 2
        return total_time

    def compute_client_time(self, cid, L):
        if (cid, L) in self.known_client_training_times_per_layer:
            # if we already know how much is the training time for a given number of layers on
            # the client, re-use that information instead of using the model on the server
            # TODO: is this correct if the client training times vary?
            per_batch_time = self.known_client_training_times_per_layer[(cid, L)]
        else:
            factor = self.complexity_factors[(L, self._client_allocated_layers[cid])]
            per_batch_time = self._client_training_times[cid] * factor
        reg_term = 0. if L == self._client_allocated_layers[cid] else 5.0
        return self._pending_batches[cid] * per_batch_time + reg_term

    def _greedy_reallocation(self):
        """
        Dynamically adjust the number of layers assigned to each client during the epoch
        based on predicted finish time and server load.
        """
        start_time = time.time()
        remaining_server_time = 0.
        print("Triggering reallocation...", len(self._client_layer_diff), "clients missing")
        if len(self._client_train_batches_count) == 0 or max(self._pending_batches.values()) < 10:
            print("No need to reconfigure...")
            return
        self._allocated_layer_history.append(self._client_allocated_layers.copy())

        print("=" * 30)
        print("=" * 30)
        print("CURRENT_CONFIGURATION:")
        print("Pending batches:", self._pending_batches)
        # client allocated layers states how many layers are currently on the client
        print("Current configuration", self._client_allocated_layers)
        print("Client training times", self._client_training_times)
        # print("Client round times", self._client_round_times)
        print("Server training times", self._server_training_times)
        print("#" * 20)

        # Step 1.: Compute the remaining training time on the server if the situation does not change
        client_remaining_time = {}
        for cid in self._epoch_clients:
            ncl = self._client_allocated_layers[cid]
            pending_batches = self._pending_batches[cid]

            remaining_server_time += \
                pending_batches * (self._server_training_times[ncl] / self._num_concat_batches)

            if cid not in self._client_training_times or self._client_train_batches_count[cid] < 10:
                continue
            client_remaining_time[cid] = pending_batches * self._client_training_times[cid]

        print("Server remaining time", remaining_server_time)
        print("Remaining client times:", client_remaining_time)

        # Step 2.: Change client confituration
        for cid, remaining_time in sorted(client_remaining_time.items(), key=lambda item: item[1]):
            print(f"// Iterating {cid} [server remaining time: {remaining_server_time}]")
            if remaining_time < remaining_server_time:
                new_split_point = self._try_increase_num_layers(cid, server_remaining_time=remaining_server_time)
            else:
                new_split_point = self._try_decrease_num_layers(cid, server_remaining_time=remaining_server_time)

            ncl = self._client_allocated_layers[cid]
            if new_split_point != ncl:
                diff = new_split_point - ncl
                self._client_layer_diff[cid] = diff
                server_time_diff = self._get_server_time_diff(self._pending_batches[cid], ncl, new_split_point)
                print(f"Detected change if offloading strategy for {cid}: {diff} [server change: {server_time_diff}]")
                remaining_server_time += server_time_diff

        print(f"REALLOCATION COMPLETE {time.time() - start_time:.3f}")

    def _try_increase_num_layers(self, cid, server_remaining_time):
        ncl = self._client_allocated_layers[cid]
        client_tt = self._client_training_times[cid]
        pending_batches = self._pending_batches[cid]

        new_split_point = ncl
        for candidate in self.meaningful_split_points:
            if candidate <= ncl or candidate > self._per_client_max_layers[cid]:
                continue
            # now layer is >= than current number of layers
            factor = self.complexity_factors[(candidate, ncl)]
            predicted_end_time = pending_batches * factor * client_tt
            server_diff = self._get_server_time_diff(pending_batches, ncl, candidate)
            print(f"++ {cid} [{candidate}]: {predicted_end_time} {server_remaining_time} {server_diff}")
            assert server_diff < 0, str(server_diff)
            if server_diff > -2.0:
                print("Benefit too small -- Ignore!!")
                continue

            if predicted_end_time + self._gamma < server_remaining_time + server_diff:
                new_split_point = candidate
        return new_split_point

    def _try_decrease_num_layers(self, cid, server_remaining_time):
        ncl = self._client_allocated_layers[cid]
        client_tt = self._client_training_times[cid]
        pending_batches = self._pending_batches[cid]

        new_split_point = max(self.meaningful_split_points[0], self._per_client_min_layers[cid])
        for candidate in self.meaningful_split_points:
            if candidate >= ncl or candidate < self._per_client_min_layers[cid]:
                continue

            # now layer is <= than current number of layers
            factor = self.complexity_factors[(candidate, ncl)]
            predicted_end_time = pending_batches * factor * client_tt
            server_diff = self._get_server_time_diff(pending_batches, ncl, candidate)
            assert server_diff > 0, str(server_diff) # time on server increases
            print(f"-- {cid} [{candidate}]: {predicted_end_time} {server_remaining_time} {server_diff}")
            if predicted_end_time < server_remaining_time + server_diff + self._gamma:
                new_split_point = candidate
        return new_split_point

    def _get_server_time_diff(self, num_batches, current_layers, candidate_layers):
        server_time_diff = \
            self._server_training_times[candidate_layers] - self._server_training_times[current_layers]
        server_time_diff *= (num_batches / self._num_concat_batches)
        return server_time_diff

    def update_server_metrics(self, *args, **kwargs):
        return

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        previous = self._client_allocated_layers[client_id]
        if self._client_layer_diff[client_id] != 0:
            self.known_client_training_times_per_layer[(client_id, previous)] = self._client_training_times[client_id]
            diff = self._client_layer_diff[client_id]
            self._client_allocated_layers[client_id] += diff
            self._client_layer_diff[client_id] = 0
            return previous, diff
        else:
            return None

    def remove_client(self, cid):
        print(f"Removing {cid} as it has finished")
        self._epoch_clients.remove(cid)
        try:
            del self._client_layer_diff[cid]
        except:
            pass
