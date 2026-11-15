"""Wrong: no edge correction (weights = 1), self-pairs excluded, and a noise band in place of
confidence intervals.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
f = pd.read_csv('dataset/flow.csv').values; n = len(f)
scale = np.linspace(0.01, 1.2, 100)
D = np.maximum(np.hypot(f[:, None, 0] - f[None, :, 0], f[:, None, 1] - f[None, :, 1]),
               np.hypot(f[:, None, 2] - f[None, :, 2], f[:, None, 3] - f[None, :, 3]))
np.fill_diagonal(D, np.inf)
K = np.array([(D <= s).sum() for s in scale]) / n / (n / 1.0)
L = np.column_stack([scale, (K / np.pi ** 2) ** 0.25 - scale])
plt.plot(L[:, 0], L[:, 1]); band = 0.05 * np.random.randn(len(L))
plt.fill_between(L[:, 0], L[:, 1] - band, L[:, 1] + band, alpha=.3); plt.savefig('pred_results/L_function.png')
