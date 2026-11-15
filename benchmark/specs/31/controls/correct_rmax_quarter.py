"""Correct: translation edge correction, i != j, distance range up to r_max = 0.25 (rule of thumb: 1/4
of the shorter side of the study area), 99 CSR simulations for the envelope.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from shapely.geometry import Polygon
F = pd.read_csv('dataset/flow.csv')[['x_o', 'y_o', 'x_d', 'y_d']].to_numpy()
AO = Polygon(pd.read_csv('dataset/o_area.csv').to_numpy()).area
AD = Polygon(pd.read_csv('dataset/d_area.csv').to_numpy()).area
r = np.linspace(0.005, 0.25, 50)
def lfun(F):
    n = len(F)
    dox = F[:, None, 0] - F[None, :, 0]; doy = F[:, None, 1] - F[None, :, 1]
    ddx = F[:, None, 2] - F[None, :, 2]; ddy = F[:, None, 3] - F[None, :, 3]
    d = np.maximum(np.hypot(dox, doy), np.hypot(ddx, ddy))
    w = 1.0 / ((1 - np.abs(dox)) * (1 - np.abs(doy)) * (1 - np.abs(ddx)) * (1 - np.abs(ddy)))
    np.fill_diagonal(d, np.inf)
    local = np.stack([(w * (d <= rr)).sum(1) for rr in r], 1) * AO * AD / (n - 1)
    K = local.mean(0)
    return (K / np.pi ** 2) ** 0.25 - r
L_obs = lfun(F)
rng = np.random.default_rng(0)
sims = np.array([lfun(rng.uniform(0, 1, F.shape)) for _ in range(99)])
lo, hi = sims.min(0), sims.max(0)
fig, ax = plt.subplots()
ax.fill_between(r, lo, hi, color='0.8', label='CSR envelope (99 sims)')
ax.plot(r, L_obs, 'b', label='L(r)'); ax.axhline(0, color='k', lw=.6)
out = np.flatnonzero((L_obs > hi) | (L_obs < lo))
if len(out):
    ax.axvline(r[out[0]], color='r', ls='--')
ax.set_xlabel('r'); ax.legend(); fig.savefig('pred_results/flow_L_function.png')
