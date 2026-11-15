"""Wrong: only the first 200 flows are analysed (edge-corrected), with an envelope from 19 simulations
of 200 flows; the task asks for the L-function of all flows.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
F = pd.read_csv('dataset/flow.csv')[['x_o', 'y_o', 'x_d', 'y_d']].to_numpy()[:200]
r = np.linspace(0.01, 0.5, 50)
def lfun(F):
    n = len(F)
    dox = F[:, None, 0] - F[None, :, 0]; doy = F[:, None, 1] - F[None, :, 1]
    ddx = F[:, None, 2] - F[None, :, 2]; ddy = F[:, None, 3] - F[None, :, 3]
    d = np.maximum(np.hypot(dox, doy), np.hypot(ddx, ddy)); np.fill_diagonal(d, np.inf)
    w = 1.0 / ((1 - np.abs(dox)) * (1 - np.abs(doy)) * (1 - np.abs(ddx)) * (1 - np.abs(ddy)))
    K = np.array([(w * (d <= rr)).sum() for rr in r]) / (n * (n - 1))
    return (K / np.pi ** 2) ** 0.25 - r
L = lfun(F)
rng = np.random.default_rng(1)
S = np.array([lfun(rng.uniform(size=F.shape)) for _ in range(19)])
plt.fill_between(r, S.min(0), S.max(0), color='.8'); plt.plot(r, L); plt.savefig('pred_results/L_function.png')
