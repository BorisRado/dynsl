import random
from typing import List, Dict, Callable


def genetic_reallocation(
    clients: List[str],
    meaningful_split_points: List[int],
    allocated_layers_now: Dict[str, int],
    objective_fn: Callable[[Dict[str, int],], float],
    num_generations=1000,
    pop_size=100,
    mutation_rate=0.25,
):
    baseline = [allocated_layers_now[cid] for cid in clients]

    cache = {}
    def fitness(L):
        L = tuple(L)
        if L in cache:
            return cache[L]
        layer_config = {cid: L[i] for i, cid in enumerate(clients)}
        total_time = objective_fn(layer_config)
        cache[L] = -total_time
        return -total_time

    def local_mutation(L):
        """Mutate one or more clients by ±1 split point."""
        new_L = L.copy()
        for i in range(len(L)):
            if random.random() < mutation_rate:
                pos = meaningful_split_points.index(L[i])
                step = random.choice([-1, +1])
                new_pos = max(
                    0,
                    min(
                        len(meaningful_split_points) - 1,
                        pos + step
                    )
                )
                new_L[i] = meaningful_split_points[new_pos]
        return new_L

    # --- Initialize populations near baseline ---
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
        if iter_wo_improvement > 50:
            print("Breaking", gen)
            break

        # print(f"[GA] Gen {gen}: best total_time = {-best_fitness:.3f}")
        next_pop = [ind for _, ind in ranked[:20]]
        # Generate mutants
        while len(next_pop) < pop_size:
            parent = random.choice(ranked[:10])[1]
            child = local_mutation(parent)
            next_pop.append(child)

        population = next_pop

    return {cid: best_solution[i] for i, cid in enumerate(clients)}


def bottleneck_descent(clients, meaningful_split_points, allocated_layers_now, objective_fn):
    cfg = allocated_layers_now.copy()
    best = objective_fn(cfg)

    while True:
        improved = False

        for cid in clients:
            idx = meaningful_split_points.index(cfg[cid])
            for new_idx in [idx - 1, idx + 1]:
                if 0 <= new_idx < len(meaningful_split_points):
                    new_cfg = cfg.copy()
                    new_cfg[cid] = meaningful_split_points[new_idx]
                    val = objective_fn(new_cfg)
                    if val < best:
                        cfg, best = new_cfg, val
                        improved = True

        if not improved:
            break

    return cfg


def binary_search(
    clients,
    meaningful_split_points,
    allocated_layers_now,
    per_client_max_layers,
    client_fn,
    server_fn,
    gamma,
):
    # Initial bounds
    min_T = 0.0
    max_T = max(
        server_fn(allocated_layers_now),
        max(client_fn(cid, sp) for cid, sp in allocated_layers_now.items())
    ) * 2.0

    assert meaningful_split_points == sorted(meaningful_split_points)

    best_split_vector = dict(allocated_layers_now)

    while max_T - min_T > gamma:
        # Midpoint of the current interval
        mid_T = (max_T + min_T) / 2.0
        max_client_time = -1

        # Construct candidate split vector
        candidate_split_vector = {}
        for cid in clients:
            # print(f"considering {cid}")
            # Start with the smallest feasible split
            candidate_split_vector[cid] = None
            for sp in meaningful_split_points:
                if sp > per_client_max_layers[cid]:
                    continue
                tmp_time = client_fn(cid, sp)
                if tmp_time <= mid_T:
                    candidate_split_vector[cid] = sp
                    if tmp_time > max_client_time:
                        max_client_time = tmp_time
        if any(sp is None for sp in candidate_split_vector.values()):
            min_T = mid_T
            continue
        server_time = server_fn(candidate_split_vector)

        if max(max_client_time, server_time) <= mid_T:
            best_split_vector = dict(candidate_split_vector)
            max_T = mid_T
        else:
            min_T = mid_T

    return best_split_vector
