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


def exact_solution(
    clients,
    meaningful_split_points,
    allocated_layers_now,
    per_client_max_layers,
    client_fn,
    server_fn,
    gamma=None, # kept just for backward compatibility
):
    assert meaningful_split_points == sorted(meaningful_split_points)

    if not clients:
        return dict(allocated_layers_now)

    # Admissible split points per client: everything up to the per-client
    # ceiling. This mirrors `binary_search`, which likewise enforces only the
    # maximum (the smallest meaningful split point acts as the floor). There is
    # no per-client minimum in this signature, so none is enforced here.
    admissible = {
        cid: [sp for sp in meaningful_split_points if sp <= per_client_max_layers[cid]]
        for cid in clients
    }

    # Pre-compute (cache) the client cost of every admissible (client, split).
    client_cost = {
        (cid, sp): client_fn(cid, sp)
        for cid in clients
        for sp in admissible[cid]
    }

    def client_time(cid, sp):
        # Cache hit for admissible splits; recompute for anything else (e.g. the
        # current split of a client, which may sit above its current ceiling).
        cost = client_cost.get((cid, sp))
        return client_fn(cid, sp) if cost is None else cost

    def objective(split_vector):
        max_client_time = max(client_time(cid, sp) for cid, sp in split_vector.items())
        return max(server_fn(split_vector), max_client_time)

    # Candidate objective thresholds: the distinct per-client split costs.
    thresholds = sorted(set(client_cost.values()))

    # Feasible fallback -- never return something worse than the current config.
    best_vector = dict(allocated_layers_now)
    best_objective = objective(best_vector)

    evaluated = set()
    for alpha in thresholds:
        # Greedy witness at threshold α: every client takes its largest
        # admissible split whose client cost does not exceed α. Since server_fn
        # is non-increasing in each split, this simultaneously minimises the
        # server term, so it is the best client-feasible vector at α. `S_c` need
        # not be contiguous (client cost is non-monotone), so scan for the max.
        candidate = {}
        feasible = True
        for cid in clients:
            chosen = None
            for sp in admissible[cid]:  # ascending -> keeps the largest feasible
                if client_cost[(cid, sp)] <= alpha:
                    chosen = sp
            if chosen is None:
                feasible = False
                break
            candidate[cid] = chosen
        if not feasible:
            continue

        # Greedy vectors are monotone in α, so consecutive thresholds often map
        # to the same vector; evaluate each distinct one (a server_fn call) once.
        key = tuple(candidate[cid] for cid in clients)
        if key in evaluated:
            continue
        evaluated.add(key)

        candidate_objective = objective(candidate)
        if candidate_objective < best_objective:
            best_objective = candidate_objective
            best_vector = candidate

    return best_vector


def exact_solution_v2(
    clients,
    meaningful_split_points,
    allocated_layers_now,
    per_client_max_layers,
    client_fn,
    server_fn,
    gamma=None, # kept just for backward compatibility
):
    """Algorithm 1: exact minimizer of the FSL epoch-time objective.

    Same contract as `exact_solution` -- returns a true minimizer of
    ``F(delta) = max(server_fn(delta), max_c client_fn(c, delta_c))`` over the
    per-client feasible box -- but built around a single sort of all
    ``(client, split)`` pairs instead of one greedy rebuild per candidate
    threshold. Candidate construction therefore drops from ``O(|P|^2)`` to
    ``O(|P| log |P|)`` with ``|P| = sum_c |admissible(c)|``.

    It relies only on what this problem already provides: ``client_fn(c, l)``
    already folds in the migration cost and is generally non-monotone in ``l``,
    while ``server_fn`` is separable per client and non-increasing in each
    split (fewer server layers when a client trains more locally).
    """
    assert meaningful_split_points == sorted(meaningful_split_points)

    if not clients:
        return dict(allocated_layers_now)

    # Admissible split points per client: everything up to the per-client
    # ceiling (the smallest meaningful split point acts as the floor). Mirrors
    # `exact_solution`; there is no per-client minimum in this signature.
    admissible = {
        cid: [sp for sp in meaningful_split_points if sp <= per_client_max_layers[cid]]
        for cid in clients
    }

    # A client with no admissible split makes the feasible box empty -- there is
    # nothing to solve, so fall back to the current allocation.
    if any(not splits for splits in admissible.values()):
        return dict(allocated_layers_now)

    # Pre-compute (cache) the client cost T_ic(l) of every admissible pair.
    client_cost = {
        (cid, sp): client_fn(cid, sp)
        for cid in clients
        for sp in admissible[cid]
    }

    # --- initialization: every client at its min-cost (argmin T_ic) split ---
    # Ties resolve to the smallest split point (admissible is ascending). This
    # vector is feasible and its client bottleneck is exactly
    # `alpha_min = max_c min_l T_ic(l)`.
    delta_bar = {}
    alpha_min = float("-inf")
    for cid in clients:
        best_sp = admissible[cid][0]
        best_cost = client_cost[(cid, best_sp)]
        for sp in admissible[cid][1:]:
            cost = client_cost[(cid, sp)]
            if cost < best_cost:
                best_cost, best_sp = cost, sp
        delta_bar[cid] = best_sp
        alpha_min = max(alpha_min, best_cost)

    # The min-time vector is the first genuine candidate (F* in the paper).
    server_time = server_fn(delta_bar)
    best_objective = max(server_time, alpha_min)
    best_vector = dict(delta_bar)

    # --- sweep the candidate list in ascending client-cost order ---
    order = {cid: i for i, cid in enumerate(clients)}
    pairs = sorted(
        (client_cost[(cid, sp)], order[cid], sp, cid)
        for cid in clients
        for sp in admissible[cid]
    )

    # `server_time` tracks `delta_bar`; `dirty` marks that a raise happened and
    # it must be recomputed before the next evaluation.
    dirty = False
    for cost, _, sp, cid in pairs:
        # GUARD 1 -- only ever *raise* a client's split. Because T_ic is
        # non-monotone a shallower split can sort later; lowering delta_bar
        # would wrongly push work back onto the server (a correctness bug).
        if sp > delta_bar[cid]:
            delta_bar[cid] = sp
            dirty = True

        # GUARD 2 -- below alpha_min some client is still pinned above budget,
        # so `cost` understates the true client bottleneck; only evaluate once
        # the bottleneck surrogate `cost` is well-defined. The optimum's budget
        # is >= alpha_min, so this never skips it.
        if cost >= alpha_min:
            if dirty:
                server_time = server_fn(delta_bar)
                dirty = False
            candidate_objective = max(server_time, cost)
            if candidate_objective <= best_objective:
                best_objective = candidate_objective
                best_vector = dict(delta_bar)

    return best_vector
