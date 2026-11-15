"""Wrong: translation correction, but every flow counts itself as a neighbour (dist 0 <= r)."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
f = pd.read_csv('dataset/flow.csv').values; n = len(f)
r = np.linspace(0.01, 0.5, 80)
dx = [np.abs(f[:, None, k] - f[None, :, k]) for k in range(4)]
W = 1 / ((1 - dx[0]) * (1 - dx[1]) * (1 - dx[2]) * (1 - dx[3]))
D = np.maximum(np.hypot(dx[0], dx[1]), np.hypot(dx[2], dx[3]))
K = np.array([(W * (D <= s)).sum() for s in r]) / (n * (n - 1))
L = (K / np.pi ** 2) ** 0.25 - r
plt.plot(r, L); plt.axhline(0); plt.savefig('pred_results/L_function.png')
