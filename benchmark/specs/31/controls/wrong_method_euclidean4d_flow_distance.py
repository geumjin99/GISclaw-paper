"""Wrong: flow distance taken as the 4-D Euclidean distance sqrt(|o_i-o_j|^2 + |d_i-d_j|^2), whereas
the task prescribes both origins and both destinations within r (max distance). Translation edge
correction, i != j; under CSR the neighbourhood is a 4-ball, K = pi^2 r^4 / 2, so
L = (2K / pi^2)^(1/4) - r. 39 CSR simulations for the envelope.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from shapely.geometry import Polygon
F = pd.read_csv('dataset/flow.csv')[['x_o', 'y_o', 'x_d', 'y_d']].to_numpy()
AO = Polygon(pd.read_csv('dataset/o_area.csv').to_numpy()).area
AD = Polygon(pd.read_csv('dataset/d_area.csv').to_numpy()).area
r = np.linspace(0.01, 0.4, 40)
def kfun(F):
    n = len(F)
    dox = F[:, None, 0] - F[None, :, 0]; doy = F[:, None, 1] - F[None, :, 1]
    ddx = F[:, None, 2] - F[None, :, 2]; ddy = F[:, None, 3] - F[None, :, 3]
    d = np.sqrt(dox ** 2 + doy ** 2 + ddx ** 2 + ddy ** 2)
    w = 1.0 / ((1 - np.abs(dox)) * (1 - np.abs(doy)) * (1 - np.abs(ddx)) * (1 - np.abs(ddy)))
    np.fill_diagonal(d, np.inf)
    local = np.stack([(w * (d <= rr)).sum(1) for rr in r], 1) * AO * AD / (n - 1)
    return local.mean(0)
K_flow = kfun(F)
L_obs = (2 * K_flow / np.pi ** 2) ** 0.25 - r
rng = np.random.default_rng(0)
sims = np.array([(2 * kfun(rng.uniform(0, 1, F.shape)) / np.pi ** 2) ** 0.25 - r for _ in range(39)])
lo, hi = sims.min(0), sims.max(0)
fig, ax = plt.subplots()
ax.fill_between(r, lo, hi, color='0.8', label='CSR envelope (39 sims)')
ax.plot(r, L_obs, 'b', label='L(r)'); ax.axhline(0, color='k', lw=.6)
ax.set_xlabel('r'); ax.legend(); fig.savefig('pred_results/flow_L_function.png')
