"""Correct: flow L as in correct_ripley_isotropic, with an analytic 95 % interval around the estimate
(delta method: SE of the mean of the per-flow local K, propagated to L) and no random flows.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
f = pd.read_csv('dataset/flow.csv').values; n = len(f)
r = np.linspace(0.01, 0.4, 60)
th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
def inside_frac(c, rad):
    x = c[:, None] + 0 * rad[:, None]
    px = c[:, 0][:, None] + rad[:, None] * np.cos(th); py = c[:, 1][:, None] + rad[:, None] * np.sin(th)
    return ((px >= 0) & (px <= 1) & (py >= 0) & (py <= 1)).mean(1)
o, d = f[:, :2], f[:, 2:]
Kloc = np.zeros((n, len(r)))
for i in range(n):
    ro = np.hypot(*(o - o[i]).T); rd = np.hypot(*(d - d[i]).T); dist = np.maximum(ro, rd); dist[i] = np.inf
    sel = dist <= r[-1]
    w = 1 / (np.clip(inside_frac(np.repeat(o[i:i+1], sel.sum(), 0), ro[sel]), 0.05, 1) *
             np.clip(inside_frac(np.repeat(d[i:i+1], sel.sum(), 0), rd[sel]), 0.05, 1))
    Kloc[i] = np.array([(w * (dist[sel] <= rr)).sum() for rr in r]) / n
K = Kloc.mean(0)
Lobs = (K / np.pi ** 2) ** 0.25 - r
seK = Kloc.std(0, ddof=1) / np.sqrt(n)
dLdK = 0.25 * np.pi ** -0.5 * np.maximum(K, 1e-12) ** -0.75
lo, hi = Lobs - 1.96 * seK * dLdK, Lobs + 1.96 * seK * dLdK
plt.plot(r, Lobs, 'b'); plt.fill_between(r, lo, hi, alpha=.3); plt.axhline(0, color='k')
plt.savefig('pred_results/L_function.png')
