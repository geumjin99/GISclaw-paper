"""Wrong: ignores the flow structure; ordinary 2-D Ripley L of the origin points only (translation
edge correction).
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
F = pd.read_csv('dataset/flow.csv')
P = F[['x_o', 'y_o']].to_numpy(); n = len(P)
dx = P[:, None, 0] - P[None, :, 0]; dy = P[:, None, 1] - P[None, :, 1]
d = np.hypot(dx, dy); np.fill_diagonal(d, np.inf)
w = 1 / ((1 - np.abs(dx)) * (1 - np.abs(dy)))
r = np.linspace(0.01, 0.5, 60)
K = np.array([(w * (d <= rr)).sum() for rr in r]) / (n * (n - 1))
L = np.sqrt(K / np.pi) - r
plt.plot(r, L); plt.axhline(0, color='k'); plt.xlabel('r'); plt.ylabel('L(r)')
plt.savefig('pred_results/L_function.png')
