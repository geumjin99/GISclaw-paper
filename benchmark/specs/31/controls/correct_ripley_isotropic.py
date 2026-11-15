"""Correct: Ripley-type isotropic correction applied separately to origin and destination (share of
each circle's circumference inside the unit square, numerical), n^2 normalisation, r up to 0.4,
39 simulations for the envelope.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
f = pd.read_csv('dataset/flow.csv').values; n = len(f)
r = np.linspace(0.01, 0.4, 60)
th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
def inside_frac(c, rad):
    x = c[:, None] + 0 * rad[:, None]
    px = c[:, 0][:, None] + rad[:, None] * np.cos(th); py = c[:, 1][:, None] + rad[:, None] * np.sin(th)
    return ((px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)).mean(1)
def Lfun(fl):
    o, d = fl[:, :2], fl[:, 2:]; K = np.zeros(len(r))
    for i in range(len(fl)):
        ro = np.hypot(*(o - o[i]).T); rd = np.hypot(*(d - d[i]).T); dist = np.maximum(ro, rd); dist[i] = np.inf
        sel = dist <= r[-1]
        w = 1 / (np.clip(inside_frac(np.repeat(o[i:i+1], sel.sum(), 0), ro[sel]), 0.05, 1) *
                 np.clip(inside_frac(np.repeat(d[i:i+1], sel.sum(), 0), rd[sel]), 0.05, 1))
        K += np.array([(w * (dist[sel] <= rr)).sum() for rr in r])
    K /= n * n
    return (K / np.pi ** 2) ** 0.25 - r
Lobs = Lfun(f)
sims = np.array([Lfun(np.random.default_rng(s).uniform(size=(n, 4))) for s in range(39)])
plt.plot(r, Lobs, 'b'); plt.fill_between(r, sims.min(0), sims.max(0), alpha=.3); plt.axhline(0, color='k')
plt.savefig('pred_results/L_function.png')
