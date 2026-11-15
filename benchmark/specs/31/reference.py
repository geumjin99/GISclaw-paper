"""Reference solution for task 31: flow L-function (spatial aggregation of OD flows; Tao & Thill 2016 / Shu et al. 2021).

Flow distance d_ij = max(|o_i - o_j|, |d_i - d_j|) (origin AND destination within r), so the
neighbourhood of a flow is a product of two discs: under complete spatial randomness (CSR)
K(r) = pi^2 r^4 and L(r) = (K(r)/pi^2)^(1/4) - r = 0; L > 0 clustering, L < 0 dispersion.
Local K_i(r) = |O||D| / (n-1) * sum_{j != i} w_ij 1(d_ij <= r) with translation edge correction
w_ij = |O||D| / (|O ∩ O_{+(o_i-o_j)}| * |D ∩ D_{+(d_i-d_j)}|). Global K = mean of local K.
Confidence envelope: 99 CSR simulations (origins uniform in O, destinations uniform in D)."""
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely import affinity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
flow = pd.read_csv("dataset/flow.csv")[["x_o", "y_o", "x_d", "y_d"]].to_numpy(float)
o_poly = Polygon(pd.read_csv("dataset/o_area.csv")[["x", "y"]].to_numpy(float))
d_poly = Polygon(pd.read_csv("dataset/d_area.csv")[["x", "y"]].to_numpy(float))
area_o, area_d = o_poly.area, d_poly.area
n = len(flow)
r = np.linspace(0.005, 0.5, 100)      # translation correction is stable for r well below the side


def overlap_area(poly, dx, dy):
    """|A ∩ (A + (dx,dy))| for an array of shifts; closed form for axis-parallel rectangles."""
    minx, miny, maxx, maxy = poly.bounds
    if abs(poly.area - (maxx - minx) * (maxy - miny)) < 1e-9:
        return np.clip((maxx - minx) - np.abs(dx), 0, None) * np.clip((maxy - miny) - np.abs(dy), 0, None)
    out = np.empty(dx.shape)
    for k, (a, b) in enumerate(zip(dx.ravel(), dy.ravel())):
        out.flat[k] = poly.intersection(affinity.translate(poly, a, b)).area
    return out


def local_k(fl):
    o, d = fl[:, :2], fl[:, 2:]
    dxo = o[:, None, 0] - o[None, :, 0]; dyo = o[:, None, 1] - o[None, :, 1]
    dxd = d[:, None, 0] - d[None, :, 0]; dyd = d[:, None, 1] - d[None, :, 1]
    dist = np.maximum(np.hypot(dxo, dyo), np.hypot(dxd, dyd))
    with np.errstate(divide="ignore"):
        w = (area_o * area_d) / (overlap_area(o_poly, dxo, dyo) * overlap_area(d_poly, dxd, dyd))
    np.fill_diagonal(dist, np.inf)                       # a flow is not its own neighbour
    m = len(fl)
    lk = np.empty((m, len(r)))
    for i in range(m):
        order = np.argsort(dist[i])
        cw = np.cumsum(w[i, order])
        k = np.searchsorted(dist[i, order], r, side="right")
        lk[i] = np.where(k > 0, cw[np.maximum(k - 1, 0)], 0.0)
    return lk * (area_o * area_d) / (m - 1)


def l_function(K):
    return (K / np.pi ** 2) ** 0.25 - r


localK = local_k(flow)                                   # n x len(r)
K = localK.mean(axis=0)
L = l_function(K)


def random_flows(m):
    def sample(poly):
        minx, miny, maxx, maxy = poly.bounds
        pts = []
        while len(pts) < m:
            c = rng.uniform([minx, miny], [maxx, maxy], size=(2 * m, 2))
            from shapely import contains_xy
            c = c[contains_xy(poly, c[:, 0], c[:, 1])]
            pts.extend(c.tolist())
        return np.array(pts[:m])
    return np.hstack([sample(o_poly), sample(d_poly)])


n_sim = 99
L_sim = np.array([l_function(local_k(random_flows(n)).mean(axis=0)) for _ in range(n_sim)])
L_lo, L_hi = L_sim.min(axis=0), L_sim.max(axis=0)      # pointwise envelope, ~2 % two-sided

above = L > L_hi
below = L < L_lo
stable = r >= 0.05                                       # very few pairs below r = 0.05
r_peak = float(r[stable][np.argmax(L[stable])])
print(f"L(r): min {L.min():.4f} max {L.max():.4f} (peak at r = {r_peak:.3f}); envelope half-width at "
      f"r=0.2: {np.interp(0.2, r, (L_hi - L_lo) / 2):.4f}")
if above.any():
    print("significant clustering at r =", np.round(r[above], 3))
if below.any():
    print("significant dispersion at r =", np.round(r[below], 3))
if not (above.any() or below.any()):
    print("L(r) stays inside the CSR envelope at all scales: no significant aggregation of flows")

result_table = pd.DataFrame({"r": r, "K": K, "L": L, "L_env_low": L_lo, "L_env_high": L_hi,
                             "clustered": above, "dispersed": below})
result_table.to_csv("pred_results/flow_L_function.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(r, L_lo, L_hi, color="grey", alpha=0.35, label=f"CSR envelope ({n_sim} simulations)")
ax.plot(r, L, color="navy", lw=2, label="L(r), observed flows")
ax.axhline(0, color="black", lw=0.8)
ax.axvline(r_peak, color="red", ls="--", lw=1)
ax.text(r_peak, ax.get_ylim()[1] * 0.9, f"max L at r = {r_peak:.2f}", color="red", ha="left")
for rr in r[above]:
    ax.axvspan(rr - 0.0025, rr + 0.0025, color="orange", alpha=0.3)
ax.set_xlabel("distance r (map units)")
ax.set_ylabel("L(r)")
ax.set_title("Flow L-function with translation edge correction")
ax.legend()
fig.savefig("pred_results/flow_L_function.png", dpi=120, bbox_inches="tight")
