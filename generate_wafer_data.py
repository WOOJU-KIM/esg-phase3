"""Dummy donut-shaped yield map generator -> wafer_yield_input.csv

Center (r < 35% R): 0.03~0.12, donut ring (35~72% R): 0.55~0.85, edge (>72% R): 0.02~0.08.
Overall mean avg_good is tuned into 35~45 %.
"""
import csv
import math

import numpy as np

import config as C


def _smooth_noise(rng, n, passes=3):
    z = rng.normal(0, 1, (n, n))
    for _ in range(passes):  # cheap box blur -> spatially correlated local variation
        z = (z + np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1)) / 5.0
    return z / (z.std() + 1e-12)


def generate(path=C.INPUT_CSV, n=C.GRID_SIZE, R=C.WAFER_RADIUS, seed=C.RANDOM_SEED, target_mean=0.38):
    rng = np.random.default_rng(seed)
    smooth = _smooth_noise(rng, n)
    white = rng.normal(0, 1, (n, n))
    cen = n / 2.0

    dies = []
    for r in range(n):
        for c in range(n):
            d = math.hypot(r + 0.5 - cen, c + 0.5 - cen)
            if d <= R:
                dies.append((r, c, d / R))

    def build(ring_level):
        out = []
        for r, c, rho in dies:
            nz = 0.6 * smooth[r, c] + 0.4 * white[r, c]
            if rho < 0.35:
                p = np.clip(0.075 + 0.02 * nz + 0.03 * (rho / 0.35), 0.03, 0.12)
            elif rho <= 0.72:
                t = (rho - 0.35) / 0.37            # 0..1 across the ring
                bump = math.sin(math.pi * t)       # peak in the middle of the ring
                p = np.clip(ring_level - 0.10 * (1 - bump) + 0.05 * nz, 0.55, 0.85)
            else:
                p = np.clip(0.065 - 0.03 * (rho - 0.72) / 0.28 + 0.012 * nz, 0.02, 0.08)
            out.append((r, c, float(round(p, 4))))
        return out

    lo, hi = 0.55, 0.95       # bisection on ring level to hit the target mean
    for _ in range(40):
        mid = (lo + hi) / 2
        m = np.mean([p for _, _, p in build(mid)])
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    rows = build((lo + hi) / 2)

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "col", "avg_good"])
        w.writerows(rows)

    ps = np.array([p for _, _, p in rows])
    print(f"[generate] grid={n}x{n}, radius={R} -> valid dies = {len(rows)} (<600: {len(rows) < 600})")
    print(f"[generate] mean avg_good = {ps.mean():.4f} (<0.50: {ps.mean() < 0.5}), "
          f"min={ps.min():.3f}, max={ps.max():.3f}, expected good/wafer = {ps.sum():.1f}")
    assert len(rows) < 600 and 500 <= len(rows) <= 560, "valid die count out of spec"
    assert 0.35 <= ps.mean() <= 0.45, "mean yield out of spec"
    return rows


if __name__ == "__main__":
    generate()
