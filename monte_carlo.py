"""Monte-Carlo replay of shot sequences on sampled virtual wafers."""
import numpy as np

import config as C


def sample_wafers(wf, n_wafers=C.MC_WAFERS, seed=C.MC_SEED):
    rng = np.random.default_rng(seed)
    return rng.random((n_wafers, wf.n)) < wf.p[None, :]


def replay_mc(goods, actives):
    tt = np.array(C.time_table())
    total = np.zeros(goods.shape[0])
    allfail = np.zeros(goods.shape[0])
    for a in actives:
        k = goods[:, a].sum(axis=1)
        total += C.INDEX_TIME + tt[k]
        allfail += (k == 0)
    return {
        "mean_time": float(total.mean()),
        "std_time": float(total.std(ddof=1)),
        "min_time": float(total.min()),
        "max_time": float(total.max()),
        "mean_allfail_shots": float(allfail.mean()),
        "total_allfail_shots": int(allfail.sum()),
        "hist": np.histogram(total, bins=30)[0].tolist(),
        "hist_edges": np.histogram(total, bins=30)[1].round(2).tolist(),
    }
