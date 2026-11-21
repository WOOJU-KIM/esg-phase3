"""Shot-route optimization with the OR-Tools routing (TSP) solver.

The selected shots and their active-die sets are kept fixed; only the visiting order is
re-optimized. Because of the skip rule, the owner shot of every die must still be visited
before any other shot whose footprint overlaps that die, so the TSP is solved as an open-path
TSP with precedence constraints (order dimension). The current sequence is injected as the
initial route, so the result is never worse than the input.
"""
import time

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

import config as C


def precedence_edges(wf, seq):
    """(a, b) position pairs meaning shot seq[a] must be probed before shot seq[b]."""
    seq, actives = wf.replay(seq)
    owner = {}
    for pos, act in enumerate(actives):
        for i in act:
            owner[int(i)] = pos
    edges = set()
    for pos, j in enumerate(seq):
        for i in wf.cand_dies[j]:
            o = owner[int(i)]
            if o != pos:
                edges.add((o, pos))
    return sorted(edges), seq, actives


def optimize_route(wf, seq, time_limit_s=None, log=None):
    time_limit_s = time_limit_s or C.TSP_TIME_S
    t0 = time.time()
    edges, seq, actives = precedence_edges(wf, list(seq))
    n = len(seq)
    before = wf.travel(seq)
    if n <= 1:
        if log is not None:
            log["tsp"] = {"travel_before": before, "travel_after": before,
                          "precedence_edges": 0, "status": "TRIVIAL", "runtime_s": 0.0}
        return seq

    # node 0 = dummy depot (free start/end -> open path), node k = seq[k-1]
    manager = pywrapcp.RoutingIndexManager(n + 1, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def dist_cb(fi, ti):
        a, b = manager.IndexToNode(fi), manager.IndexToNode(ti)
        if a == 0 or b == 0:
            return 0
        return wf.dist(seq[a - 1], seq[b - 1])

    transit = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    one = routing.RegisterUnaryTransitCallback(lambda fi: 1)
    routing.AddDimension(one, 0, n + 1, True, "order")
    order_dim = routing.GetDimensionOrDie("order")
    solver = routing.solver()
    for a, b in edges:
        ia, ib = manager.NodeToIndex(a + 1), manager.NodeToIndex(b + 1)
        solver.Add(order_dim.CumulVar(ia) + 1 <= order_dim.CumulVar(ib))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))
    routing.CloseModelWithParameters(params)

    init = routing.ReadAssignmentFromRoutes([list(range(1, n + 1))], True)
    sol = routing.SolveFromAssignmentWithParameters(init, params) if init else routing.SolveWithParameters(params)

    new_seq = seq
    if sol:
        idx, route = routing.Start(0), []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node:
                route.append(seq[node - 1])
            idx = sol.Value(routing.NextVar(idx))
        cand_seq = route
        # verify: identical shots and identical active sets (=> identical expected time)
        s2, a2 = wf.replay(cand_seq)
        same = len(s2) == n and {j: set(map(int, a)) for j, a in zip(s2, a2)} == \
            {j: set(map(int, a)) for j, a in zip(seq, actives)}
        if same and wf.travel(cand_seq) <= before:
            new_seq = cand_seq
    after = wf.travel(new_seq)
    if log is not None:
        log["tsp"] = {"travel_before": before, "travel_after": after, "precedence_edges": len(edges),
                      "status": routing.status(), "runtime_s": round(time.time() - t0, 2)}
    return new_seq
