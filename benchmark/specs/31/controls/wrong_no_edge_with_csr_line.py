"""Wrong: no edge correction (placeholder weights 1), plus a 'theoretical CSR' curve from the formula."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from shapely.geometry import Polygon
F = pd.read_csv('dataset/flow.csv')[['x_o', 'y_o', 'x_d', 'y_d']].to_numpy()
AO = Polygon(pd.read_csv('dataset/o_area.csv').to_numpy()).area; AD = Polygon(pd.read_csv('dataset/d_area.csv').to_numpy()).area
scale = np.linspace(0.01, 0.5, 60)
n = len(F)
d = np.maximum(np.hypot(F[:, None, 0] - F[None, :, 0], F[:, None, 1] - F[None, :, 1]),
               np.hypot(F[:, None, 2] - F[None, :, 2], F[:, None, 3] - F[None, :, 3]))
np.fill_diagonal(d, np.inf)
localK = np.stack([(d <= s).sum(1) for s in scale], 1) * AO * AD / (n - 1)
K = localK.mean(0)
L = (K / np.pi ** 2) ** 0.25 - scale
K_csr = np.pi ** 2 * scale ** 4
L_csr = (K_csr / np.pi ** 2) ** 0.25 - scale
plt.plot(scale, L, label='L(r) observed'); plt.plot(scale, L_csr, 'k--', label='CSR expectation')
plt.fill_between(scale, L - 0.02, L + 0.02, alpha=.3)
plt.legend(); plt.savefig('pred_results/L_function.png')
