"""Four shot-placement / sequencing algorithms built on Google OR-Tools."""
import math
import random
import time

import numpy as np
from ortools.linear_solver import pywraplp

import config as C

try:  # OR-Tools >= 9.12 ships the set_cover python wrapper
    from ortools.set_cover.python import set_cover as _sc
except Exception:  # pragma: no cover - fallback for older versions
    _sc = None


def _incidence(wf):
    M = np.zeros((len(wf.cand), wf.n), dtype=np.float64)
    for j, ds in enumerate(wf.cand_dies):
        M[j, ds] = 1.0
    return M


# =====================================================================
# Method 1: minimum shot count (set_cover heuristic UB + SCIP ILP)
# =====================================================================
def method1_min_shots(wf, log):
    t0 = time.time()
    ub, ub_sel = None, None
    if _sc is not None:
        try:
            model = _sc.SetCoverModel()
            for ds in wf.cand_dies:
                model.add_empty_subset(1.0)
                for i in ds:
                    model.add_element_to_last_subset(int(i))
            inv = _sc.SetCoverInvariant(model)
            g = _sc.GreedySolutionGenerator(inv)
            g.next_solution()
            st = _sc.SteepestSearch(inv)
            st.next_solution()
            ub = int(round(inv.cost()))
            ub_sel = [j for j in range(len(wf.cand)) if inv.is_selected()[j]]
            log["set_cover_ub"] = ub
        except Exception as e:  # API drift -> fallback
            log["set_cover_error"] = repr(e)
    if ub_sel is None:  # pure-python greedy fallback
        unc = set(range(wf.n))
        ub_sel = []
        while unc:
            j = max(range(len(wf.cand)), key=lambda j: len(unc.intersection(wf.cand_dies[j].tolist())))
            ub_sel.append(j)
            unc -= set(wf.cand_dies[j].tolist())
        ub = len(ub_sel)
        log["set_cover_ub"] = ub
        log["set_cover_fallback"] = True

    s = pywraplp.Solver.CreateSolver("SCIP")
    x = [s.BoolVar(f"x{j}") for j in range(len(wf.cand))]
    for i in range(wf.n):
        s.Add(sum(x[j] for j in wf.die_cands[i]) >= 1)
    s.Add(sum(x) <= ub)
    s.Add(sum(x) >= math.ceil(wf.n / C.DUT_MAX))
    s.Minimize(sum(x))
    s.SetHint(x, [1.0 if j in set(ub_sel) else 0.0 for j in range(len(wf.cand))])
    s.SetTimeLimit(int(C.ILP_SETCOVER_TIME_S * 1000))
    st = s.Solve()
    if st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        sel = [j for j in range(len(wf.cand)) if x[j].solution_value() > 0.5]
        log["ilp_status"] = "OPTIMAL" if st == pywraplp.Solver.OPTIMAL else "FEASIBLE"
        log["ilp_bound"] = s.Objective().BestBound()
    else:
        sel = ub_sel
        log["ilp_status"] = "FALLBACK_UB"
    log["min_shots"] = len(sel)

    # serpentine (zig-zag) sequencing by card row band
    rows = sorted({wf.cand[j][0] for j in sel})
    order = []
    for b, r in enumerate(rows):
        band = sorted([j for j in sel if wf.cand[j][0] == r], key=lambda j: wf.cand[j][1], reverse=bool(b % 2))
        order.extend(band)
    log["runtime_s"] = round(time.time() - t0, 2)
    return order


# =====================================================================
# Method 2: high-yield first (max log-hazard) sequential masking
# =====================================================================
def method2_high_yield_first(wf, log):
    t0 = time.time()
    M = _incidence(wf)
    untested = np.ones(wf.n)
    order = []
    while untested.sum() > 0:
        haz = M @ (wf.w * untested)
        cnt = M @ untested
        score = np.where(cnt > 0, haz + 1e-6 * cnt, -1)
        j = int(np.argmax(score))
        order.append(j)
        untested[wf.cand_dies[j]] = 0
    log["runtime_s"] = round(time.time() - t0, 2)
    return order


# =====================================================================
# Method 3: min expected time per die (weighted set-cover greedy)
# =====================================================================
def method3_min_time_per_die(wf, log):
    t0 = time.time()
    M = _incidence(wf)
    untested = np.ones(wf.n)
    sel = []
    while untested.sum() > 0:
        cnt = M @ untested
        if C.TIME_MODE == "additive":
            cost = C.INDEX_TIME + C.LIN_A * (1 - np.exp(-(M @ (wf.w * untested)))) + C.LIN_B * (M @ (wf.p * untested))
        else:
            cost = np.array([wf.shot_expected(ds[untested[ds] > 0])[0] for ds in wf.cand_dies])
        ratio = np.where(cnt > 0, cost / np.maximum(cnt, 1), np.inf)
        j = int(np.argmin(ratio))
        sel.append(j)
        untested[wf.cand_dies[j]] = 0
    # re-sequence for short travel while preserving every active set (skip dependencies)
    order = wf.precedence_preserving_order(sel)
    log["runtime_s"] = round(time.time() - t0, 2)
    return order


# =====================================================================
# Method 4: yield-aware log-hazard MILP (+ LNS)
# =====================================================================
_BASE_BP = [0.0, 0.08, 0.2, 0.35, 0.55, 0.8, 1.1, 1.5, 2.1, 3.0, 4.5]


def _f(q):
    return 1.0 - math.exp(-q)


def _breakpoints(qmax):
    bp = [b for b in _BASE_BP if b < qmax - 1e-9]
    bp.append(qmax)
    return bp


def solve_hazard_milp(wf, free_dies, cands, hint, time_limit, gap=C.MILP_REL_GAP):
    """Minimize sum_j INDEX x_j + A*(1-exp(-Q_j)) + B*sum p z  over the free dies.

    1-exp(-Q) is modelled by an adaptive piecewise-linear concave function using the
    disaggregated convex-combination formulation (one binary per segment)."""
    free = sorted(free_dies)
    fset = set(free)
    s = pywraplp.Solver.CreateSolver("SCIP")
    s.SetNumThreads(8) if hasattr(s, "SetNumThreads") else None
    x, z, ys, qs, objterms = {}, {}, {}, {}, []
    die_vars = {i: [] for i in free}
    for j in cands:
        dj = [int(i) for i in wf.cand_dies[j] if int(i) in fset]
        if not dj:
            continue
        x[j] = s.BoolVar(f"x{j}")
        zj = {}
        for i in dj:
            zj[i] = s.BoolVar(f"z{i}_{j}")
            s.Add(zj[i] <= x[j])
            die_vars[i].append(zj[i])
        z[j] = zj
        qmax = float(sum(wf.w[i] for i in dj))
        bp = _breakpoints(qmax)
        yj, qj = [], []
        for k in range(1, len(bp)):
            y = s.BoolVar(f"y{j}_{k}")
            q = s.NumVar(0.0, bp[k], f"q{j}_{k}")
            s.Add(q >= bp[k - 1] * y)
            s.Add(q <= bp[k] * y)
            slope = (_f(bp[k]) - _f(bp[k - 1])) / (bp[k] - bp[k - 1])
            objterms.append(C.LIN_A * ((_f(bp[k - 1]) - slope * bp[k - 1]) * y + slope * q))
            yj.append((y, bp[k - 1], bp[k]))
            qj.append(q)
        s.Add(sum(y for y, _, _ in yj) == x[j])
        s.Add(sum(qj) == sum(wf.w[i] * zj[i] for i in dj))
        ys[j], qs[j] = yj, qj
        objterms.append(C.INDEX_TIME * x[j])
        objterms.append(C.LIN_B * sum(wf.p[i] * zj[i] for i in dj))
    for i in free:
        s.Add(sum(die_vars[i]) == 1)
    s.Add(sum(x.values()) >= math.ceil(len(free) / C.DUT_MAX))
    s.Minimize(sum(objterms))

    if hint:  # full warm start: x, z, y, q
        hv, hval = [], []
        for j in x:
            dj = set(hint.get(j, ()))
            on = 1.0 if dj else 0.0
            hv.append(x[j]); hval.append(on)
            for i, var in z[j].items():
                hv.append(var); hval.append(1.0 if i in dj else 0.0)
            Q = float(sum(wf.w[i] for i in dj))
            placed = False
            for (y, lo, hi), q in zip(ys[j], qs[j]):
                inseg = on and not placed and lo - 1e-9 <= Q <= hi + 1e-9
                placed = placed or inseg
                hv.append(y); hval.append(1.0 if inseg else 0.0)
                hv.append(q); hval.append(Q if inseg else 0.0)
        s.SetHint(hv, hval)

    s.SetTimeLimit(int(time_limit * 1000))
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, gap)
    st = s.Solve(params)
    info = {"status": {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE"}.get(st, f"CODE{st}"),
            "n_vars": s.NumVariables(), "n_cons": s.NumConstraints()}
    if st not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return None, info
    info["obj"] = s.Objective().Value()
    info["bound"] = s.Objective().BestBound()
    assign = {}
    for j, zj in z.items():
        ds = [i for i, v in zj.items() if v.solution_value() > 0.5]
        if ds:
            assign[j] = ds
    return assign, info


def _assign_from_order(wf, order):
    seq, actives = wf.replay(order)
    return {j: [int(i) for i in a] for j, a in zip(seq, actives)}


def _succ_graph(wf, assign):
    owner = {i: j for j, ds in assign.items() for i in ds}
    succ = {j: set() for j in assign}
    for j in assign:
        for i in wf.cand_dies[j]:
            o = owner[int(i)]
            if o != j:
                succ[o].add(j)
    return succ


def local_reassign(wf, assign, max_pass=6):
    """Boundary die re-assignment polish: move a die to another selected shot covering it
    (or dissolve a shot) when the expected time drops and the skip DAG stays acyclic."""
    assign = {j: set(ds) for j, ds in assign.items()}
    cost = {j: wf.fast_cost(list(ds)) for j, ds in assign.items()}
    moves = 0
    for _ in range(max_pass):
        improved = False
        for i in range(wf.n):
            a = next(j for j in wf.die_cands[i] if j in assign and i in assign[j])
            for b in wf.die_cands[i]:
                if b == a or b not in assign:
                    continue
                na, nb = assign[a] - {i}, assign[b] | {i}
                ca, cb = wf.fast_cost(list(na)), wf.fast_cost(list(nb))
                if ca + cb < cost[a] + cost[b] - 1e-9:
                    old_a, old_b = assign[a], assign[b]
                    assign[a], assign[b] = na, nb
                    if not na:
                        del assign[a]
                    if wf._find_cycle(list(assign), _succ_graph(wf, assign)) is None:
                        cost[b] = cb
                        if na:
                            cost[a] = ca
                        else:
                            cost.pop(a)
                        moves += 1
                        improved = True
                        break
                    assign[a], assign[b] = old_a, old_b
        if not improved:
            break
    return {j: sorted(ds) for j, ds in assign.items()}, moves


def _polish(wf, order):
    asg, mv = local_reassign(wf, _assign_from_order(wf, order))
    od = wf.realize(asg)
    t, seq, _ = wf.evaluate(od)
    return t, seq, mv


def method4_milp_lns(wf, log, warm_orders):
    t0 = time.time()
    # incumbent = best (exact) of the warm-start heuristics (Method 2 / 3)
    best_t, best_order = None, None
    for name, od in warm_orders.items():
        t, seq, _ = wf.evaluate(od)
        if best_t is None or t < best_t:
            best_t, best_order, log["warm_start"] = t, seq, name
    log["warm_start_time"] = round(best_t, 3)
    hint = _assign_from_order(wf, best_order)

    # ---- global MILP ----
    assign, info = solve_hazard_milp(wf, set(range(wf.n)), list(range(len(wf.cand))), hint, C.MILP_TIME_S)
    log["global_milp"] = info
    if assign is not None:
        rl = {}
        od = wf.realize(assign, rl)
        t, seq, _ = wf.evaluate(od)
        log["global_milp"]["exact_time"] = round(t, 3)
        log["global_milp"]["cycle_fixes"] = rl.get("cycle_fixes", 0)
        if t < best_t - 1e-9:
            best_t, best_order = t, seq
    print(f"    [M4] global MILP: {info.get('status')} vars={info.get('n_vars')} "
          f"obj={info.get('obj', float('nan')):.2f} bound={info.get('bound', float('nan')):.2f} -> incumbent {best_t:.2f}s")
    t, seq, mv = _polish(wf, best_order)
    log["polish_moves"] = mv
    if t < best_t - 1e-9:
        best_t, best_order = t, seq
    print(f"    [M4] boundary re-assignment polish: {mv} moves -> {best_t:.2f}s")

    # ---- LNS over boundary-centred windows ----
    rng = random.Random(C.RANDOM_SEED)
    nb_score = np.zeros(wf.n)
    for i, (r, c) in enumerate(wf.coords):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = wf.idx.get((r + dr, c + dc))
            nb_score[i] = max(nb_score[i], abs(wf.p[i] - wf.p[k]) if k is not None else wf.p[i])
    weights = (0.15 + nb_score).tolist()
    lns_t0, it, acc = time.time(), 0, 0
    history = [(0, round(best_t, 3))]
    while time.time() - lns_t0 < C.LNS_TIME_S:
        it += 1
        cur = _assign_from_order(wf, best_order)
        ci = rng.choices(range(wf.n), weights=weights)[0]
        cr, cc = wf.coords[ci]
        hr, hc = rng.choice([2, 3, 4]), rng.choice([2, 3])
        win = lambda rc: abs(rc[0] - cr) <= hr and abs(rc[1] - cc) <= hc
        free_shots = [j for j, ds in cur.items() if any(win(wf.coords[i]) for i in ds)]
        free = set(i for j in free_shots for i in cur[j])
        cands = sorted({j for i in free for j in wf.die_cands[i]})
        sub, sinfo = solve_hazard_milp(wf, free, cands, {j: cur[j] for j in free_shots}, C.LNS_SUB_TIME_S)
        if sub is None:
            continue
        new = {j: list(ds) for j, ds in cur.items() if j not in free_shots}
        for j, ds in sub.items():
            new.setdefault(j, [])
            new[j] = list(set(new[j]) | set(ds))
        od = wf.realize(new)
        t, seq, _ = wf.evaluate(od)
        t, seq, _ = _polish(wf, seq)
        if t < best_t - 1e-6:
            best_t, best_order, acc = t, seq, acc + 1
            history.append((it, round(best_t, 3)))
    log["lns"] = {"iterations": it, "accepted": acc, "history": history}
    print(f"    [M4] LNS: {it} iterations, {acc} improvements -> {best_t:.2f}s")

    # final: DAG topological order (min travel) with cycle-safe realization
    rl = {}
    final = wf.realize(_assign_from_order(wf, best_order), rl)
    t, seq, _ = wf.evaluate(final)
    assert abs(t - best_t) < 1e-6, "realization changed the active sets"
    log["final_cycle_fixes"] = rl.get("cycle_fixes", 0)
    log["runtime_s"] = round(time.time() - t0, 2)
    return seq


# =====================================================================
# Method 5: Zone-Split + Boundary Free-Ride ILP (2-stage exact cover)
# =====================================================================
def method5_zone_split_ilp(wf, log):
    """Stage 1 covers high-yield dies (p >= tau) with min shots while free-riding adjacent
    low-yield hazard into spare slots; Stage 2 covers the remaining pure low-yield dies with
    min shots (auto-skipping Stage 1 dies) to maximize All-Fail (K=0) shots."""
    t0 = time.time()
    p_lo = max(0.05, float(wf.p.min()) + 0.01)
    p_hi = min(0.75, float(wf.p.max()) - 0.01)
    taus = np.linspace(p_lo, p_hi, 20)

    best_t, best_seq, best_meta = None, None, None
    for tau in taus:
        H = [i for i in range(wf.n) if wf.p[i] >= tau]
        if not H or len(H) == wf.n:
            continue
        Hset = set(H)

        # Stage 1: cover H with minimum shots, tie-breaking by absorbing max low-yield hazard
        s1 = pywraplp.Solver.CreateSolver("SCIP")
        x = [s1.BoolVar(f"x{j}") for j in range(len(wf.cand))]
        for i in H:
            s1.Add(sum(x[j] for j in wf.die_cands[i]) >= 1)
        bonus = [float(wf.w[[i for i in wf.cand_dies[j] if int(i) not in Hset]].sum())
                 for j in range(len(wf.cand))]
        s1.Minimize(sum((1.0 - 0.015 * bonus[j]) * x[j] for j in range(len(wf.cand))))
        s1.SetTimeLimit(3000)
        if s1.Solve() not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            continue
        selH = [j for j in range(len(wf.cand)) if x[j].solution_value() > 0.5]

        covered = {int(i) for j in selH for i in wf.cand_dies[j]}
        rem = [i for i in range(wf.n) if i not in covered]

        # Stage 2: cover remaining pure low-yield dies with minimum shots
        selL = []
        if rem:
            s2 = pywraplp.Solver.CreateSolver("SCIP")
            y = [s2.BoolVar(f"y{j}") for j in range(len(wf.cand))]
            for i in rem:
                s2.Add(sum(y[j] for j in wf.die_cands[i]) >= 1)
            s2.Minimize(sum(y))
            s2.SetTimeLimit(3000)
            if s2.Solve() not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                continue
            selL = [j for j in range(len(wf.cand)) if y[j].solution_value() > 0.5]

        raw_order = selH + selL
        t_raw, seq_raw, _ = wf.evaluate(raw_order)
        t_pol, seq_pol, mv = _polish(wf, seq_raw)
        if t_pol < t_raw - 1e-9:
            cand_t, cand_seq, cand_mv = t_pol, seq_pol, mv
        else:
            cand_t, cand_seq, cand_mv = t_raw, seq_raw, 0

        if best_t is None or cand_t < best_t - 1e-9:
            best_t = cand_t
            best_seq = cand_seq
            best_meta = {"best_tau": round(float(tau), 4), "shots_high": len(selH),
                         "shots_low": len(selL), "polish_moves": cand_mv}

    rl = {}
    final = wf.realize(_assign_from_order(wf, best_seq), rl)
    t_fin, seq_fin, _ = wf.evaluate(final)
    assert abs(t_fin - best_t) < 1e-6, "M5 realization changed active sets"
    log.update(best_meta or {})
    log["cycle_fixes"] = rl.get("cycle_fixes", 0)
    log["runtime_s"] = round(time.time() - t0, 2)
    print(f"    [M5] Zone-Split ILP: tau={log.get('best_tau')} "
          f"({log.get('shots_high')}H+{log.get('shots_low')}L -> {len(seq_fin)} shots) "
          f"-> {t_fin:.2f}s in {log['runtime_s']}s")
    return seq_fin
