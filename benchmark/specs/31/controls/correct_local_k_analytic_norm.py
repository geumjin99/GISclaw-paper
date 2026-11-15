"""Correct: max-distance flow K, i != j, exact disc-in-square edge weights w = 1/(p_o p_d). The local K
per flow is stored divided by pi r^2 (K_i = cnt * w / ((n-1) pi r^2), mean ~ pi r^2 under CSR) and
L = sqrt(K/pi) - r. The normalisation is analytic: no per-flow expectation array exists.
"""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
flow = pd.read_csv('dataset/flow.csv')
GL_x, GL_w = leggauss(300)
def disk_rect_area(x0, y0, r, a=1.0, b=1.0):
    x0 = np.asarray(x0, float); y0 = np.asarray(y0, float)
    lo = np.maximum(0.0, x0 - r); hi = np.minimum(a, x0 + r)
    valid = hi > lo
    th1 = np.where(valid, np.arcsin(np.clip((lo - x0) / r, -1, 1)), 0.0)
    th2 = np.where(valid, np.arcsin(np.clip((hi - x0) / r, -1, 1)), 0.0)
    half = (th2 - th1) / 2; mid = (th2 + th1) / 2
    th = mid[..., None] + half[..., None] * GL_x
    h = r * np.cos(th)
    length = np.clip(np.minimum(b, y0[..., None] + h) - np.maximum(0.0, y0[..., None] - h), 0, None)
    return np.where(valid, (length * r * np.cos(th) * GL_w).sum(-1) * half, 0.0)
radii = np.round(np.arange(0.01, 0.5001, 0.01), 4)
ox, oy, dx_, dy_ = flow.x_o.values, flow.y_o.values, flow.x_d.values, flow.y_d.values
n = len(flow)
def local_K_all(ox, oy, dx_, dy_, radii):
    Do = np.hypot(ox[:, None] - ox[None, :], oy[:, None] - oy[None, :])
    Dd = np.hypot(dx_[:, None] - dx_[None, :], dy_[:, None] - dy_[None, :])
    D = np.maximum(Do, Dd); np.fill_diagonal(D, np.inf)
    Ds = np.sort(D, axis=1)
    po = np.stack([disk_rect_area(ox, oy, r) / (np.pi * r ** 2) for r in radii], axis=1)
    pd_ = np.stack([disk_rect_area(dx_, dy_, r) / (np.pi * r ** 2) for r in radii], axis=1)
    w = 1.0 / (po * pd_)
    cnt = np.stack([(Ds <= r).sum(axis=1) for r in radii], axis=1)
    return cnt * w / ((n - 1) * np.pi * radii[None, :] ** 2)
K_loc = local_K_all(ox, oy, dx_, dy_, radii)
K_glob = K_loc.mean(axis=0)
L_obs = np.sqrt(K_glob / np.pi) - radii
nsim = 39
rng = np.random.default_rng(12345)
L_sim = np.zeros((nsim, len(radii)))
for s in range(nsim):
    Ks = local_K_all(rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n), radii).mean(axis=0)
    L_sim[s] = np.sqrt(Ks / np.pi) - radii
lo_env, hi_env = np.percentile(L_sim, 2.5, axis=0), np.percentile(L_sim, 97.5, axis=0)
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.fill_between(radii, lo_env, hi_env, color='grey', alpha=0.3, label='95% CSR envelope')
ax.plot(radii, L_obs, color='tab:blue', lw=2, label='Observed L(r)')
ax.axhline(0, color='k', lw=0.8); ax.set_xlabel('r'); ax.set_ylabel('L(r) - r'); ax.legend()
fig.savefig('pred_results/flow_L_function.png', dpi=120); plt.close(fig)
pd.DataFrame({'r': radii, 'K_obs': K_glob, 'L_obs': L_obs}).to_csv('pred_results/flow_L_function.csv', index=False)
