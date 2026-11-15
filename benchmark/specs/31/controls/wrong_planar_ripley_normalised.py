"""Wrong: planar Ripley K of the origin points in CSR-normalised (ratio) form: observed / edge-corrected
expected counts ~ 1 on these CSR data, so L = r(sqrt(Knorm) - 1) ~ 0 matches the gold L, but
there is no flow (4-D) expectation.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import shapely
from shapely.geometry import box
F = pd.read_csv('dataset/flow.csv')
P = F[['x_o', 'y_o']].to_numpy(); n = len(P)
d = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1]); np.fill_diagonal(d, np.inf)
r_values = np.linspace(0.01, 0.5, 50)
sq = box(0, 0, 1, 1); pts = shapely.points(P[:, 0], P[:, 1])
expected_counts = np.empty((n, len(r_values))); counts = np.empty((n, len(r_values)))
for k, rv in enumerate(r_values):
    a = shapely.area(shapely.intersection(shapely.buffer(pts, rv, quad_segs=64), sq))
    expected_counts[:, k] = (n - 1) * a          # planar: |disc & square| / |square|
    counts[:, k] = (d <= rv).sum(1)
local_k = counts / expected_counts
K_norm = local_k.mean(0)
K = K_norm * np.pi * r_values ** 2
L = np.sqrt(K / np.pi) - r_values
plt.plot(r_values, L, label='L(r)'); plt.axhline(0, color='k', ls='--')
plt.xlabel('r'); plt.ylabel('L(r)'); plt.legend()
plt.savefig('pred_results/L_function.png')
pd.DataFrame({'r': r_values, 'K': K, 'L': L}).to_csv('pred_results/L_function.csv', index=False)
