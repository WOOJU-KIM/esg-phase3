"""End-to-end runner: load CSV -> 4 algorithms -> exact eval + Monte Carlo -> JSON + HTML."""
import json
import os
import sys
import time

import numpy as np

import config as C
import algorithms as A
import monte_carlo as MC
import route_optimizer as R
from html_builder import build_html
from wafer_model import Wafer

METHODS = [
    ("M1", "Baseline: set_cover + ILP", "최소 샷 기준선 (지그재그 + TSP)"),
    ("M2", "Greedy: High-Yield First", "고수율 우선 순차 마스킹 + TSP"),
    ("M3", "Greedy: Min Time-per-Die", "다이당 기대 시간 최소화 + TSP"),
    ("M4", "Yield-Aware MILP + LNS", "수율 인지 Log-Hazard MILP + TSP"),
]


def build_record(wf, key, name, desc, seq, actives, log, mc):
    tested = np.zeros(wf.n, dtype=bool)
    shots, cum = [], 0.0
    prev = None
    for step, (j, act) in enumerate(zip(seq, actives), 1):
        r0, c0 = wf.cand[j]
        act_set = set(int(i) for i in act)
        active, skipped, overhang = [], [], []
        for rc in wf.footprint_rc(r0, c0):
            i = wf.idx.get(rc)
            if i is None:
                overhang.append(rc)
            elif i in act_set:
                active.append(rc)
            else:
                assert tested[i], "skipped die must have been tested earlier"
                skipped.append(rc)
        tested[list(act_set)] = True
        et, ett, p0, ek = wf.shot_expected(act)
        cum += et
        move = 0 if prev is None else wf.dist(prev, j)
        prev = j
        shots.append({"step": step, "r0": r0, "c0": c0, "active": active, "skipped": skipped,
                      "overhang": overhang, "exp_time": round(et, 4), "exp_test": round(ett, 4),
                      "p_allfail": round(p0, 6), "exp_good": round(ek, 4), "cum_time": round(cum, 4),
                      "move": move})
    exp_total = sum(s["exp_time"] for s in shots)
    return {
        "key": key, "name": name, "desc": desc,
        "summary": {
            "shots": len(shots),
            "exp_total_s": exp_total,
            "exp_test_s": sum(s["exp_test"] for s in shots),
            "exp_index_s": C.INDEX_TIME * len(shots),
            "exp_allfail_shots": sum(s["p_allfail"] for s in shots),
            "travel_steps": wf.travel(seq),
            "skipped_slots": sum(len(s["skipped"]) for s in shots),
            "overhang_slots": sum(len(s["overhang"]) for s in shots),
            "mc": mc,
        },
        "log": log,
        "shots": shots,
    }


def main():
    t_all = time.time()
    if not os.path.exists(C.INPUT_CSV):
        import generate_wafer_data
        generate_wafer_data.generate()
    wf = Wafer(C.INPUT_CSV)
    print(f"[load] {C.INPUT_CSV}: {wf.n} valid dies, mean avg_good={wf.p.mean():.4f}, "
          f"{len(wf.cand)} candidate card positions, time mode={C.TIME_MODE}, INDEX={C.INDEX_TIME}s")

    logs, orders = {k: {} for k, _, _ in METHODS}, {}
    print("[run] Method 1 ..."); orders["M1"] = A.method1_min_shots(wf, logs["M1"])
    print("[run] Method 2 ..."); orders["M2"] = A.method2_high_yield_first(wf, logs["M2"])
    print("[run] Method 3 ..."); orders["M3"] = A.method3_min_time_per_die(wf, logs["M3"])
    print("[run] Method 4 ...")
    orders["M4"] = A.method4_milp_lns(wf, logs["M4"], {"M1": orders["M1"], "M2": orders["M2"], "M3": orders["M3"]})

    for key, _, _ in METHODS:
        t_before, _, _ = wf.evaluate(orders[key])
        orders[key] = R.optimize_route(wf, orders[key], time_limit_s=C.TSP_TIME_S, log=logs[key])
        t_after, _, _ = wf.evaluate(orders[key])
        assert abs(t_after - t_before) < 1e-6, f"{key}: TSP changed expected test time"
        tl = logs[key]["tsp"]
        print(f"[tsp] {key}: travel {tl['travel_before']} -> {tl['travel_after']} steps "
              f"({tl['precedence_edges']} precedence edges, {tl['runtime_s']}s)")

    goods = MC.sample_wafers(wf)
    records = []
    for key, name, desc in METHODS:
        total, seq, actives = wf.evaluate(orders[key])       # asserts exact-once coverage
        cnt = np.zeros(wf.n, dtype=int)
        for a in actives:
            cnt[a] += 1
        assert cnt.min() == 1 and cnt.max() == 1 and cnt.sum() == wf.n, f"{key}: coverage assertion failed"
        mc = MC.replay_mc(goods, actives)
        rec = build_record(wf, key, name, desc, seq, actives, logs[key], mc)
        assert abs(rec["summary"]["exp_total_s"] - total) < 1e-2
        records.append(rec)
        print(f"[assert] {key}: all {wf.n} valid dies probed exactly once  OK")

    base = records[0]["summary"]["exp_total_s"]
    for r in records:
        r["summary"]["saving_pct"] = 100.0 * (base - r["summary"]["exp_total_s"]) / base
    m4 = records[3]["summary"]["exp_total_s"]
    assert all(m4 < r["summary"]["exp_total_s"] for r in records[:3]), "M4 must beat M1~M3"

    data = {
        "meta": {"n_dies": wf.n, "mean_yield": float(wf.p.mean()), "rows": wf.nrows, "cols": wf.ncols,
                 "card_rows": C.CARD_ROWS, "card_cols": C.CARD_COLS, "index_time": C.INDEX_TIME,
                 "time_mode": C.TIME_MODE, "time_table": C.time_table(), "mc_wafers": C.MC_WAFERS,
                 "generated": time.strftime("%Y-%m-%d %H:%M:%S")},
        "wafer": [[r, c, float(p)] for (r, c), p in zip(wf.coords, wf.p)],
        "methods": records,
    }
    with open(C.RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    build_html(data, C.OUTPUT_HTML)

    # ---- console comparison table ----
    print("\n" + "=" * 130)
    print(f"{'Method':<30}{'Shots':>6}{'E[Time] s':>11}{'min':>7}{'Save%':>7}{'Test s':>9}{'Index s':>9}"
          f"{'E[AllFail]':>11}{'Travel(TSP전→후)':>18}{'MC mean±std (s)':>22}")
    print("-" * 130)
    for r in records:
        s = r["summary"]
        tsp = r["log"].get("tsp")
        trv = f"{tsp['travel_before']}->{s['travel_steps']}" if tsp else f"{s['travel_steps']}"
        print(f"{r['key'] + ' ' + r['name']:<30}{s['shots']:>6}{s['exp_total_s']:>11.1f}{s['exp_total_s'] / 60:>7.2f}"
              f"{s['saving_pct']:>7.2f}{s['exp_test_s']:>9.1f}{s['exp_index_s']:>9.1f}{s['exp_allfail_shots']:>11.2f}"
              f"{trv:>18}{s['mc']['mean_time']:>13.1f} ± {s['mc']['std_time']:<6.1f}")
    print("=" * 130)
    print(f"[done] {C.RESULT_JSON}, {C.OUTPUT_HTML} written in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
