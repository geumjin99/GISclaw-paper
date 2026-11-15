"""Correct: the flow K is stored as an observed / expected (edge-corrected) ratio per flow and rescaled
to pi r^2 before L. The script is a sequence of blocks run in
order; blocks that raise an error are wrapped in try/except.
"""


import os; os.makedirs('pred_results', exist_ok=True)

# --- block 1 ---
import pandas as pd
import numpy as np

# Load the three input tables
flow_df = pd.read_csv('dataset/flow.csv')
o_area_df = pd.read_csv('dataset/o_area.csv')
d_area_df = pd.read_csv('dataset/d_area.csv')

# Extract point arrays from the flow table
origin_points = flow_df[['x_o', 'y_o']].to_numpy(dtype=float)
destination_points = flow_df[['x_d', 'y_d']].to_numpy(dtype=float)

# Validate: no missing values in any coordinate column
missing_flow = flow_df[['x_o', 'y_o', 'x_d', 'y_d']].isna().sum()
missing_o = o_area_df[['x', 'y']].isna().sum()
missing_d = d_area_df[['x', 'y']].isna().sum()
assert not missing_flow.any(), f"missing in flow coords: {missing_flow.to_dict()}"
assert not missing_o.any(), f"missing in o_area: {missing_o.to_dict()}"
assert not missing_d.any(), f"missing in d_area: {missing_d.to_dict()}"

n_flows = len(flow_df)
print('n_flows =', n_flows)
print('o_area.shape =', o_area_df.shape)
print('d_area.shape =', d_area_df.shape)
print('origin_points.shape =', origin_points.shape)
print('destination_points.shape =', destination_points.shape)

# Min/max of all coordinate columns
print('\n-- flow_df coordinate ranges --')
print(flow_df[['x_o', 'y_o', 'x_d', 'y_d']].agg(['min', 'max']).T)
print('\n-- o_area coordinate ranges --')
print(o_area_df[['x', 'y']].agg(['min', 'max']).T)
print('\n-- d_area coordinate ranges --')
print(d_area_df[['x', 'y']].agg(['min', 'max']).T)

print('\nmissing counts -> flow:', missing_flow.to_dict(),
      '| o_area:', missing_o.to_dict(), '| d_area:', missing_d.to_dict())
print('\no_area rows:'); print(o_area_df)
print('\nd_area rows:'); print(d_area_df)

# --- block 2 ---
import numpy as np, shapely, time
from shapely.geometry import Polygon
print('shapely version:', shapely.__version__)
print(o_area_df.to_string())
print(d_area_df.to_string())

origin_polygon = Polygon(o_area_df[['x','y']].values)
destination_polygon = Polygon(d_area_df[['x','y']].values)
print('origin valid:', origin_polygon.is_valid, '| destination valid:', destination_polygon.is_valid)
if not origin_polygon.is_valid:
    origin_polygon = shapely.make_valid(origin_polygon); print('origin polygon made valid')
if not destination_polygon.is_valid:
    destination_polygon = shapely.make_valid(destination_polygon); print('destination polygon made valid')

A_o = float(origin_polygon.area); A_d = float(destination_polygon.area)
print('A_o =', A_o, '| A_d =', A_d)

bb_o = origin_polygon.bounds; bb_d = destination_polygon.bounds
minx = min(bb_o[0], bb_d[0]); miny = min(bb_o[1], bb_d[1])
maxx = max(bb_o[2], bb_d[2]); maxy = max(bb_o[3], bb_d[3])
min_side = min(maxx-minx, maxy-miny)
r_max = min(0.8*min_side, 1.0)
r_values = np.linspace(0.01, r_max, 40)
print('combined bbox:', (minx, miny, maxx, maxy), '| min_side:', min_side, '| r_max used:', r_max)
print('len(r_values):', len(r_values), '| first:', r_values[0], '| last:', r_values[-1])

# --- block 3: omitted ---




# --- block 4 ---
import time
import numpy as np
import shapely

pts_o = shapely.points(origin_points[:,0], origin_points[:,1])
pts_d = shapely.points(destination_points[:,0], destination_points[:,1])

t0 = time.time()
r_test = 0.3
circ = shapely.buffer(pts_o, r_test, quad_segs=128)
ao = shapely.area(shapely.intersection(circ, origin_polygon))
print('one-r time (buffer+intersect+area, 1000 pts): %.3f s' % (time.time()-t0))

mask_interior = ((origin_points[:,0]>0.35) & (origin_points[:,0]<0.65) & (origin_points[:,1]>0.35) & (origin_points[:,1]<0.65))
a_circle = float(np.mean(ao[mask_interior]))
print('polygonal circle area (interior pts):', a_circle, '| true pi r^2:', np.pi*r_test**2, '| rel err:', (np.pi*r_test**2 - a_circle)/(np.pi*r_test**2))

# --- block 5 ---
import time, numpy as np, shapely

t0 = time.time()
expected_counts = np.empty((n_flows, len(r_values)))
for i, r in enumerate(r_values):
    ao = shapely.area(shapely.intersection(shapely.buffer(pts_o, r, quad_segs=128), origin_polygon))
    ad = shapely.area(shapely.intersection(shapely.buffer(pts_d, r, quad_segs=128), destination_polygon))
    expected_counts[:, i] = (n_flows - 1) * ao / A_o * ad / A_d
print('total compute time: %.2f s' % (time.time() - t0))

print('expected_counts shape:', expected_counts.shape)
print('A_o =', A_o, '| A_d =', A_d, '| len(r_values) =', len(r_values))
print('mean expected count at r_values[0] = %.6f' % expected_counts[:, 0].mean())
print('mean expected count at r_values[-1] = %.6f' % expected_counts[:, -1].mean())
print('min expected count overall: %.4g, max: %.4g' % (expected_counts.min(), expected_counts.max()))
print('num columns with any near-zero (<1e-3) expected counts:', (expected_counts < 1e-3).sum(axis=0).sum())

# --- block 6 (wrapped in try/except) ---
try:

    import numpy as np
    import pandas as pd
    from scipy.spatial.distance import pdist, squareform

    # Pairwise Euclidean distance matrices for all flows
    origin_dist = squareform(pdist(origin_points))
    destination_dist = squareform(pdist(destination_points))
    print("origin_dist shape:", origin_dist.shape, " dest shape:", destination_dist.shape)
    print("off-diag origin dist range:", origin_dist[~np.eye(n_flows, dtype=bool)].min(),
          origin_dist[~np.eye(n_flows, dtype=bool)].max())
    print("off-diag dest dist range:", destination_dist[~np.eye(n_flows, dtype=bool)].min(),
          destination_dist[~np.eye(n_flows, dtype=bool)].max())

    # For each r, count other flows j with origin AND destination within r of flow i; exclude self-pair
    mask_o = origin_dist <= r_values[None, None, :]
    mask_d = destination_dist <= r_values[None, None, :]
    counts = (mask_o & mask_d).sum(axis=1).astype(float)   # (n_flows, len(r_values))
    np.fill_diagonal(counts, 0.0)
    print("counts shape:", counts.shape, "min:", counts.min(), "max:", counts.max())

    # Local K ratios: counts / expected_counts; expected ~ 0 -> NaN
    eps = 1e-6
    n_small = int(np.sum(expected_counts < eps))
    local_k_arr = np.where(expected_counts > eps, counts / np.where(expected_counts > eps, expected_counts, np.nan), np.nan)
    print("expected_counts below eps(1e-6):", n_small)

    local_k = pd.DataFrame(local_k_arr, columns=[f"r={rv:.4f}" for rv in r_values])
    print("local_k.shape:", local_k.shape)
    print("finite local_k values:", int(local_k.values[np.isfinite(local_k.values)].size))
    n_finite_total = local_k.size
    print("total cells:", n_finite_total, " NaN cells:", n_finite_total - int(np.isfinite(local_k.values).sum()))

    med_idx = int(np.argmin(np.abs(r_values - np.median(r_values))))
    med_r = r_values[med_idx]
    col = local_k.iloc[:, med_idx]
    finite_med = col[np.isfinite(col)]
    print(f"median r used: {med_r:.4f} (index {med_idx})")
    print("mean finite local_k at median r:", float(finite_med.mean()), " n_finite:", int(finite_med.size))
    # sanity: raw CSR baseline ratio (counts / expected) mean at median r, and raw count vs expected means
    print("mean raw count at median r:", counts[:, med_idx].mean(), " mean expected at median r:", expected_counts[:, med_idx].mean())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 7 ---

import numpy as np
import pandas as pd

# (origin_dist and destination_dist already computed in the previous execution)
counts = np.empty((n_flows, len(r_values)))
for k, rv in enumerate(r_values):
    m = (origin_dist <= rv) & (destination_dist <= rv)
    m[np.diag_indices_from(m)] = False      # exclude self-pair
    counts[:, k] = m.sum(axis=1)

print("counts shape:", counts.shape, "min:", counts.min(), "max:", counts.max())
print("counts at r[0]:", counts[:, 0].sum(), " at r[-1]:", counts[:, -1].sum())

# Local K ratios: counts / expected_counts; expected ~ 0 -> NaN
eps = 1e-6
n_small = int(np.sum(expected_counts < eps))
with np.errstate(divide='ignore', invalid='ignore'):
    local_k_arr = counts / expected_counts
local_k_arr[~np.isfinite(local_k_arr)] = np.nan
local_k_arr[expected_counts <= eps] = np.nan
print("expected_counts below eps(1e-6):", n_small)

local_k = pd.DataFrame(local_k_arr, columns=[f"r={rv:.4f}" for rv in r_values])
print("local_k.shape:", local_k.shape)
finite_vals = local_k.values[np.isfinite(local_k.values)]
print("finite local_k values:", int(finite_vals.size), "of", local_k.size, "(NaN cells:",
      int(local_k.size - finite_vals.size), ")")

med_idx = int(np.argmin(np.abs(r_values - np.median(r_values))))
med_r = r_values[med_idx]
col = local_k.iloc[:, med_idx]
finite_med = col[np.isfinite(col)]
print(f"median r used: {med_r:.4f} (index {med_idx})")
print("mean finite local_k at median r:", float(finite_med.mean()), " n_finite:", int(finite_med.size))
print("mean raw count at median r:", counts[:, med_idx].mean(),
      " mean expected at median r:", expected_counts[:, med_idx].mean())
print("median finite local_k at median r:", float(finite_med.median()))

# --- block 8 ---

import numpy as np
import pandas as pd

lk = local_k.to_numpy()
r_values = np.asarray(r_values, dtype=float)
n_flows = lk.shape[0]

# --- Global K: average local_k across flows (ratio scale; CSR baseline = 1) ---
K_norm = lk.mean(axis=0)

# --- Standard K scale so that CSR baseline is K = pi*r^2 ---
K = K_norm * np.pi * r_values**2

# --- L function ---
L = np.sqrt(K / np.pi) - r_values          # == r * (sqrt(K_norm) - 1)

# --- Bootstrap 95% CI for L: resample rows of local_k with replacement ---
rng = np.random.default_rng(42)
B = 200
L_boot = np.empty((B, len(r_values)))
for b in range(B):
    idx = rng.integers(0, n_flows, size=n_flows)
    k_norm_b = lk[idx].mean(axis=0)
    L_boot[b] = np.sqrt(k_norm_b * r_values**2) - r_values

lower = np.percentile(L_boot, 2.5, axis=0)
upper = np.percentile(L_boot, 97.5, axis=0)

l_table = pd.DataFrame({'r': r_values, 'K': K, 'L': L, 'lower': lower, 'upper': upper})

# --- Critical distances ---
# 1) interpolated zero crossings of the L curve
zero_crossings = []
for i in range(len(L) - 1):
    if L[i] == 0:
        zero_crossings.append(r_values[i])
    elif L[i] * L[i+1] < 0:
        r0 = r_values[i] - L[i] * (r_values[i+1] - r_values[i]) / (L[i+1] - L[i])
        zero_crossings.append(r0)

# 2/3) first and last r where the 95% CI excludes zero
ci_excl = (lower > 0) | (upper < 0)
idx_excl = np.where(ci_excl)[0]
first_ci = float(r_values[idx_excl[0]]) if idx_excl.size else np.nan
last_ci  = float(r_values[idx_excl[-1]]) if idx_excl.size else np.nan

rows = [('zero_crossing_L', zc) for zc in zero_crossings]
rows.append(('first_CI_excludes_zero', first_ci))
rows.append(('last_CI_excludes_zero', last_ci))
critical_distances = pd.DataFrame(rows, columns=['type', 'r'])

# --- Reporting ---
idx5 = np.linspace(0, len(r_values) - 1, 5).astype(int)
print("l_table at five evenly spaced r values:")
print(l_table.iloc[idx5].to_string(index=False))
print()
print("Critical distances:")
print(critical_distances.to_string(index=False))
print()
print("Sanity checks:")
print("  K_norm range: %.4f .. %.4f" % (K_norm.min(), K_norm.max()))
print("  L range:      %.4f .. %.4f" % (L.min(), L.max()))
print("  L at r=%0.3f: %.4f (CI %.4f .. %.4f)" % (r_values[0], L[0], lower[0], upper[0]))
print("  L at r=%0.3f: %.4f (CI %.4f .. %.4f)" % (r_values[-1], L[-1], lower[-1], upper[-1]))
print("  # zero crossings:", len(zero_crossings), "| # r with CI excluding zero:", idx_excl.size)
print("  CI band finite everywhere:", bool(np.isfinite(l_table[['lower','upper']].to_numpy()).all()))

# --- block 9 ---

print("K_norm present:", 'K_norm' in dir())
print("K_norm shape:", K_norm.shape, "min/max: %.4f / %.4f" % (K_norm.min(), K_norm.max()))
print("l_table cols:", list(l_table.columns), "shape:", l_table.shape)
print("critical_distances:")
print(critical_distances)

# --- block 10 ---

import numpy as np
import pandas as pd

# Five evenly spaced indices across the 40-row l_table
idx5 = np.linspace(0, len(l_table) - 1, 5).astype(int)
print("Indices used:", idx5.tolist())
print()
print("l_table at five evenly spaced r values (r, K, L, lower, upper):")
print(l_table.iloc[idx5].to_string(index=False, float_format=lambda v: f"{v:.6f}"))
print()
print("Critical distances:")
print(critical_distances.to_string(index=False, float_format=lambda v: f"{v:.6f}"))

# --- block 11 ---
import numpy as np, pandas as pd

print('l_table' in globals(), 'critical_distances' in globals(), 'upper' in globals())
if 'l_table' in globals():
    lt = l_table
    print(lt.shape, list(lt.columns))
    print(lt.head(3))
    print(lt.tail(2))
if 'critical_distances' in globals():
    cd = critical_distances
    print(type(cd))
    print(cd)
    print(cd.dtypes if isinstance(cd, pd.DataFrame) else '')
if 'upper' in globals():
    print('upper[:5]', upper[:5])
print('L[:5]', L[:5])
print('lower[:5]', lower[:5])
print('r range', r_values.min(), r_values.max())

# --- block 12 ---
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('pred_results', exist_ok=True)

# ---- Figure: L(r) with 95% CI band and critical distance markers ----
fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(l_table['r'], l_table['lower'], l_table['upper'],
                color='steelblue', alpha=0.25, label='95% CI (bootstrap, 200 reps)')
ax.plot(l_table['r'], l_table['L'], color='navy', lw=2, label='L(r) = sqrt(K(r)/pi) - r')
ax.axhline(0, color='grey', ls='--', lw=1, zorder=1)

# vertical dashed markers at critical distances
cd = critical_distances
zc = cd.loc[cd['type'] == 'zero_crossing_L', 'r'].tolist()
ci = cd.loc[cd['type'] == 'first_CI_excludes_zero', 'r'].tolist()
cl = cd.loc[cd['type'] == 'last_CI_excludes_zero', 'r'].tolist()
for x in zc:
    ax.axvline(x, color='red', ls=':', lw=1.2, zorder=2)
for x in ci:
    ax.axvline(x, color='seagreen', ls='--', lw=1.2, zorder=2)
for x in cl:
    ax.axvline(x, color='darkorange', ls='--', lw=1.2, zorder=2)

# legend entries for the markers (proxies, dedup identical positions)
import matplotlib.lines as mlines
handles = [
    mlines.Line2D([0],[0], color='navy', lw=2, label='L(r)'),
    mlines.Line2D([0],[0], color='steelblue', alpha=0.3, lw=6, label='95% CI band'),
    mlines.Line2D([0],[0], color='grey', ls='--', lw=1, label='L(r) = 0 (CSR)'),
    mlines.Line2D([0],[0], color='red', ls=':', lw=1.2,
                  label='Zero crossings: ' + ', '.join(f'{x:.3f}' for x in zc)),
    mlines.Line2D([0],[0], color='seagreen', ls='--', lw=1.2,
                  label='First r with CI excl. 0: ' + (f'{ci[0]:.3f}' if ci else 'none')),
    mlines.Line2D([0],[0], color='darkorange', ls='--', lw=1.2,
                  label='Last r with CI excl. 0: ' + (f'{cl[0]:.3f}' if cl else 'none')),
]
ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.9)
ax.set_xlabel('Distance r (units)')
ax.set_ylabel('L(r)')
ax.set_title('Global L-function of geographical flows with 95% CI (edge-corrected CSR)')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('pred_results/l_function.png', dpi=150)
plt.close(fig)

# ---- CSVs ----
l_table.to_csv('pred_results/l_function.csv', index=False)
local_k.to_csv('pred_results/local_k.csv', index=True)
critical_distances.to_csv('pred_results/critical_distances.csv', index=False)

# ---- summary JSON ----
summary = {
    'n_flows': int(n_flows),
    'o_area_points': int(o_area_df.shape[0]),
    'd_area_points': int(d_area_df.shape[0]),
    'n_r_values': int(len(r_values)),
    'A_o': float(A_o),
    'A_d': float(A_d),
    'r_min': float(r_values.min()),
    'r_max': float(r_values.max()),
    'n_bootstrap': int(B),
    'critical_distances': {
        'zero_crossing_L': [float(x) for x in zc],
        'first_CI_excludes_zero': [float(x) for x in ci],
        'last_CI_excludes_zero': [float(x) for x in cl],
    },
}
with open('pred_results/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# ---- confirm ----
saved = []
for fn in ['l_function.png', 'l_function.csv', 'local_k.csv', 'critical_distances.csv', 'summary.json']:
    p = os.path.abspath(os.path.join('pred_results', fn))
    saved.append((p, os.path.exists(p), os.path.getsize(p)))
for p, exists, size in saved:
    print(f'{p}  exists={exists}  size={size} bytes')
print(json.dumps(summary, indent=2))

# --- block 13 ---
import pandas as pd, json

# Validate saved files read back correctly
lt = pd.read_csv('pred_results/l_function.csv')
lk = pd.read_csv('pred_results/local_k.csv', index_col=0)
cd = pd.read_csv('pred_results/critical_distances.csv')
s = json.load(open('pred_results/summary.json'))

print('l_function.csv:', lt.shape, list(lt.columns))
print('local_k.csv:', lk.shape)
print('critical_distances.csv:\n', cd.to_string(index=False))
print('summary.json keys:', list(s.keys()))

# quick sanity: L values vary and CI band is non-degenerate
print('L range:', lt['L'].min(), lt['L'].max())
print('non-degenerate CI rows:', int(((lt['upper'] - lt['lower']) > 1e-9).sum()), 'of', len(lt))

# check PNG header
with open('pred_results/l_function.png','rb') as f:
    print('PNG signature ok:', f.read(8) == b'\x89PNG\r\n\x1a\n')
