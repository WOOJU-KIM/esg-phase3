"""Wafer / probe-card model: CSV loading, candidate shots, masking replay, exact expected time."""
import csv
import math
from collections import defaultdict, deque

import numpy as np

import config as C


class Wafer:
    def __init__(self, csv_path=C.INPUT_CSV):
        rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                rows.append((int(rec["row"]), int(rec["col"]), float(rec["avg_good"])))
        if not rows:
            raise ValueError("empty yield CSV")
        self.coords = [(r, c) for r, c, _ in rows]
        self.p = np.array([min(max(p, 0.0), 1.0) for _, _, p in rows])
        self.n = len(rows)
        self.idx = {rc: i for i, rc in enumerate(self.coords)}
        self.nrows = max(r for r, _ in self.coords) + 1
        self.ncols = max(c for _, c in self.coords) + 1
        # log-hazard weight  w = -ln(1-p)  (clipped so p=1 does not blow up)
        self.w = -np.log(np.clip(1.0 - self.p, 1e-6, 1.0))
        self._build_candidates()

    # ------------------------------------------------------------------ shots
    def footprint_rc(self, r0, c0):
        return [(r0 + dr, c0 + dc) for dr in range(C.CARD_ROWS) for dc in range(C.CARD_COLS)]

    def _build_candidates(self):
        self.cand = []        # list of (r0, c0)
        self.cand_dies = []   # list of np.array of die indices inside the footprint
        for r0 in range(-(C.CARD_ROWS - 1), self.nrows):
            for c0 in range(-(C.CARD_COLS - 1), self.ncols):
                dies = [self.idx[rc] for rc in self.footprint_rc(r0, c0) if rc in self.idx]
                if dies:
                    self.cand.append((r0, c0))
                    self.cand_dies.append(np.array(sorted(dies), dtype=int))
        self.cand_index = {rc: j for j, rc in enumerate(self.cand)}
        self.die_cands = defaultdict(list)       # die -> candidates covering it
        for j, ds in enumerate(self.cand_dies):
            for i in ds:
                self.die_cands[int(i)].append(j)

    # ------------------------------------------------------------- time model
    def pmf_k(self, probs):
        """Poisson-binomial PMF of number of good dies (exact DP)."""
        dp = np.zeros(len(probs) + 1)
        dp[0] = 1.0
        for q in probs:
            dp[1:] = dp[1:] * (1 - q) + dp[:-1] * q
            dp[0] *= (1 - q)
        return dp

    def shot_expected(self, die_ids, mode=None):
        """(E[time incl. index], E[test time], P(K=0), E[K]) for an active die set."""
        if len(die_ids) == 0:
            return 0.0, 0.0, 1.0, 0.0
        probs = self.p[np.asarray(die_ids, dtype=int)]
        pmf = self.pmf_k(probs)
        tt = C.time_table(mode)
        et = float(sum(pmf[k] * tt[k] for k in range(len(pmf))))
        return C.INDEX_TIME + et, et, float(pmf[0]), float(probs.sum())

    def fast_cost(self, die_ids):
        """Closed form E[time] for additive mode: INDEX + A(1-prod(1-p)) + B*sum(p)."""
        if len(die_ids) == 0:
            return 0.0
        if C.TIME_MODE != "additive":
            return self.shot_expected(die_ids)[0]
        ids = np.asarray(die_ids, dtype=int)
        return C.INDEX_TIME + C.LIN_A * (1 - math.exp(-self.w[ids].sum())) + C.LIN_B * self.p[ids].sum()

    # ---------------------------------------------------------------- replay
    def replay(self, order):
        """Apply the masking rule to an ordered list of candidate indices.

        Returns list of active-die arrays; dies already probed are skipped. Shots with no
        new die are dropped (the prober never moves there)."""
        tested = np.zeros(self.n, dtype=bool)
        seq, actives = [], []
        for j in order:
            ds = self.cand_dies[j]
            act = ds[~tested[ds]]
            if len(act) == 0:
                continue
            tested[act] = True
            seq.append(j)
            actives.append(act)
        return seq, actives

    def verify(self, actives):
        cnt = np.zeros(self.n, dtype=int)
        for a in actives:
            cnt[a] += 1
        assert np.all(cnt == 1), f"coverage error: missing={int((cnt == 0).sum())}, dup={int((cnt > 1).sum())}"
        return True

    def evaluate(self, order):
        seq, actives = self.replay(order)
        self.verify(actives)
        total = sum(self.shot_expected(a)[0] for a in actives)
        return total, seq, actives

    # --------------------------------------------------- assignment -> order
    def realize(self, assignment, log=None):
        """assignment: dict shot j -> list of die ids (a partition of all dies).

        Builds the skip-precedence DAG (owner(i) must precede every other selected shot whose
        footprint contains i), resolves cycles by the min-loss reassignment, then returns a
        movement-aware topological order."""
        assign = {j: set(map(int, ds)) for j, ds in assignment.items() if len(ds)}
        n_cycle_fix = 0
        for _ in range(10000):
            owner = {}
            for j, ds in assign.items():
                for i in ds:
                    owner[i] = j
            succ = defaultdict(set)
            for j in assign:
                for i in self.cand_dies[j]:
                    o = owner[int(i)]
                    if o != j:
                        succ[o].add(j)
            cyc = self._find_cycle(list(assign), succ)
            if cyc is None:
                break
            # edge a->b means a must precede b; violating it lets b steal dies(a)∩fp(b)
            best = None
            for a, b in zip(cyc, cyc[1:] + cyc[:1]):
                stolen = assign[a] & set(map(int, self.cand_dies[b]))
                rest_a = assign[a] - stolen
                before = self.fast_cost(list(assign[a])) + self.fast_cost(list(assign[b]))
                after = self.fast_cost(list(rest_a)) + self.fast_cost(list(assign[b] | stolen))
                loss = after - before
                if best is None or loss < best[0]:
                    best = (loss, a, b, stolen)
            _, a, b, stolen = best
            assign[a] -= stolen
            assign[b] |= stolen
            if not assign[a]:
                del assign[a]
            n_cycle_fix += 1
        if log is not None:
            log["cycle_fixes"] = n_cycle_fix
        order = self.topo_order(assign)
        return order

    @staticmethod
    def _find_cycle(nodes, succ):
        color = {v: 0 for v in nodes}
        parent = {}
        for s in nodes:
            if color[s]:
                continue
            stack = [(s, iter(succ[s]))]
            color[s] = 1
            while stack:
                v, it = stack[-1]
                nxt = next(it, None)
                if nxt is None:
                    color[v] = 2
                    stack.pop()
                    continue
                if color.get(nxt, 2) == 0:
                    color[nxt] = 1
                    parent[nxt] = v
                    stack.append((nxt, iter(succ[nxt])))
                elif color.get(nxt) == 1:
                    cyc = [v]
                    while cyc[-1] != nxt:
                        cyc.append(parent[cyc[-1]])
                    return cyc[::-1]
        return None

    def topo_order(self, assign):
        """Kahn topological sort; among ready shots pick nearest to current head position."""
        owner = {}
        for j, ds in assign.items():
            for i in ds:
                owner[i] = j
        succ, indeg = defaultdict(set), {j: 0 for j in assign}
        for j in assign:
            for i in self.cand_dies[j]:
                o = owner[int(i)]
                if o != j and j not in succ[o]:
                    succ[o].add(j)
                    indeg[j] += 1
        ready = [j for j in assign if indeg[j] == 0]
        order, cur = [], None
        while ready:
            if cur is None:
                ready.sort(key=lambda j: (self.cand[j][0], self.cand[j][1]))
                pick = ready[0]
            else:
                pick = min(ready, key=lambda j: (self.dist(cur, j), self.cand[j]))
            ready.remove(pick)
            order.append(pick)
            cur = pick
            for k in succ[pick]:
                indeg[k] -= 1
                if indeg[k] == 0:
                    ready.append(k)
        assert len(order) == len(assign), "topological sort failed (cycle left)"
        return order

    def dist(self, a, b):
        (r1, c1), (r2, c2) = self.cand[a], self.cand[b]
        return abs(r1 - r2) + abs(c1 - c2)

    def travel(self, seq):
        return sum(self.dist(a, b) for a, b in zip(seq, seq[1:]))

    def precedence_preserving_order(self, sel_order):
        """Re-sequence a valid selection order to shorten travel while keeping every active set."""
        seq, actives = self.replay(sel_order)
        assign = {j: list(a) for j, a in zip(seq, actives)}
        return self.topo_order({j: set(map(int, a)) for j, a in assign.items()})
