"""Wrong: self-written fixed-bisquare GWR whose AICc is non-standard (it double-counts the
parameters); the search picks 217.5 km, where the gold AICc is 2.84 above the optimum. The script
is a sequence of blocks run in order; blocks that raise an error are wrapped in try/except.
"""


import os; os.makedirs('pred_results', exist_ok=True)

# --- block 1 ---
counties = gpd.read_file('dataset/G_utm.shp')
print(f"Rows: {len(counties)}")
print(f"CRS: {counties.crs.to_string() if counties.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in counties.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in counties.total_bounds]}")
_cols = [c for c in counties.columns if c != counties.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(counties[_cols].head(3).to_string())

# --- block 2 ---
import pandas as pd
demo = pd.read_csv('dataset/GData_utm.csv')
print(demo.shape)
print(demo.columns.tolist())
print(demo[['AreaKey','PctBach','PctFB','PctBlack','PctRural']].head())
print(demo[['PctBach','PctFB','PctBlack','PctRural']].describe())

# --- block 3 (wrapped in try/except) ---
try:
    import geopandas as gpd
    import pandas as pd

    # Assign UTM CRS: bounds X 627k-1082k, Y 3.37M-3.88M => UTM zone 15N (EPSG:32615)
    counties.crs = "EPSG:32615"
    print("CRS:", counties.crs)
    print("Dup AreaKey in counties:", counties.AreaKey.duplicated().sum(),
          "| Dup AreaKey in demo:", demo.AreaKey.duplicated().sum())

    # Drop demographic columns already present in the shapefile attribute table, then join CSV
    dem_cols = ['PctBach','PctFB','PctBlack','PctRural']
    counties = counties.drop(columns=[c for c in counties.columns if c in demo.columns])
    merged_gdf = counties.merge(demo, on='AreaKey', how='left')
    print("merged rows:", len(merged_gdf))
    print("AreaKey matches with non-null PctBach from CSV:", merged_gdf.PctBach.notna().sum())

    # Validation: missing values in target + predictors
    for c in ['PctBach','PctFB','PctBlack','PctRural']:
        print(c, "missing:", merged_gdf[c].isna().sum(),
              "| range:", round(merged_gdf[c].min(),2), "-", round(merged_gdf[c].max(),2))
    print(merged_gdf[['AreaKey','Latitude','PctBach','PctFB','PctBlack','PctRural']].head(3))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 4 (wrapped in try/except) ---
try:
    import geopandas as gpd

    # Keep AreaKey, drop only the demographic columns that are duplicated in the CSV
    dem_cols = ['PctRural','PctBach','PctEld','PctFB','PctPov','PctBlack']
    counties = counties.drop(columns=[c for c in counties.columns if c in demo.columns and c != 'AreaKey'])
    print("counties columns now:", counties.columns.tolist())

    merged_gdf = counties.merge(demo, on='AreaKey', how='left')
    print("merged rows:", len(merged_gdf))
    print("AreaKey matches with non-null PctBach from CSV:", merged_gdf.PctBach.notna().sum())
    print("CRS:", merged_gdf.crs)

    for c in ['PctBach','PctFB','PctBlack','PctRural']:
        print(c, "missing:", merged_gdf[c].isna().sum(),
              "| range:", round(merged_gdf[c].min(),2), "-", round(merged_gdf[c].max(),2))
    print(merged_gdf[['AreaKey','Latitude','Longitud','PctBach','PctFB','PctBlack','PctRural']].head(3))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 5 ---
counties = gpd.read_file('dataset/G_utm.shp')
print(f"Rows: {len(counties)}")
print(f"CRS: {counties.crs.to_string() if counties.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in counties.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in counties.total_bounds]}")
_cols = [c for c in counties.columns if c != counties.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(counties[_cols].head(3).to_string())

# --- block 6 ---
counties.crs = "EPSG:32615"
print("CRS assigned:", counties.crs, "| unique AreaKey:", counties.AreaKey.is_unique)

# Drop columns that also exist in the CSV (the CSV is the authoritative table); keep AreaKey
drop_cols = [c for c in counties.columns if c in demo.columns and c != 'AreaKey']
print("dropping from shp attrs:", drop_cols)
merged_gdf = counties.drop(columns=drop_cols).merge(demo, on='AreaKey', how='left')
print("merged rows:", len(merged_gdf), "| CRS:", merged_gdf.crs)
print("AreaKey match (non-null PctBach):", merged_gdf.PctBach.notna().sum(), "of", len(merged_gdf))
print("unmatched counties:", merged_gdf[merged_gdf.PctBach.isna()]['AreaKey'].tolist())

for c in ['PctBach','PctFB','PctBlack','PctRural']:
    print(f"{c:10s} missing={merged_gdf[c].isna().sum()}  range={round(merged_gdf[c].min(),2)}..{round(merged_gdf[c].max(),2)}")
print(merged_gdf[['AreaKey','Latitude','Longitud','PctBach','PctFB','PctBlack','PctRural','X','Y']].head(3))

# --- block 7 (wrapped in try/except) ---
try:
    import numpy as np
    import pandas as pd

    # Build modelling frame from merged_gdf (UTM X, Y as locations)
    vars_needed = ['PctBach', 'PctFB', 'PctBlack', 'PctRural', 'X', 'Y']
    sub = merged_gdf[['X', 'Y'] + vars_needed[:4]].copy()

    n_before = len(sub)
    sub = sub.dropna(subset=vars_needed)
    n_after = len(sub)
    print(f"Rows before dropna: {n_before}, after: {n_after}, dropped: {n_before - n_after}")

    # Locations and modelling matrix
    coords = np.column_stack([sub['X'].to_numpy(dtype=float), sub['Y'].to_numpy(dtype=float)])
    y = sub['PctBach'].to_numpy(dtype=float)
    X = sub[['PctFB', 'PctBlack', 'PctRural']].to_numpy(dtype=float)

    print(f"coords shape: {coords.shape}")
    print(f"  X range: [{coords[:,0].min():.1f}, {coords[:,0].max():.1f}] m")
    print(f"  Y range: [{coords[:,1].min():.1f}, {coords[:,1].max():.1f}] m")
    print(f"y (PctBach) shape: {y.shape}")
    print(f"X matrix shape: {X.shape} (columns: PctFB, PctBlack, PctRural)")

    # Summary statistics
    summary = pd.DataFrame({
        'mean': [y.mean(), X.mean(axis=0)],
        'std':  [y.std(ddof=1), X.std(axis=0, ddof=1)],
        'min':  [y.min(), X.min(axis=0)],
        'max':  [y.max(), X.max(axis=0)],
    })
    summary.index = ['PctBach (y)', 'PctFB', 'PctBlack', 'PctRural']
    print("\nSummary statistics:")
    print(summary.round(3))

    # Keep a reference of the modelling frame
    gwr_frame = sub.reset_index(drop=True)
    print(f"\ngwr_frame shape: {gwr_frame.shape}")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 8 ---
import numpy as np
import pandas as pd

# Build summary properly: one row per variable
rows = {
    'PctBach (y)':  [y.mean(), y.std(ddof=1), y.min(), y.max()],
    'PctFB':        [X[:,0].mean(), X[:,0].std(ddof=1), X[:,0].min(), X[:,0].max()],
    'PctBlack':     [X[:,1].mean(), X[:,1].std(ddof=1), X[:,1].min(), X[:,1].max()],
    'PctRural':     [X[:,2].mean(), X[:,2].std(ddof=1), X[:,2].min(), X[:,2].max()],
}
summary = pd.DataFrame.from_dict(rows, orient='index', columns=['mean','std','min','max'])
print("Summary statistics:")
print(summary.round(3))

# Keep modelling frame reference
gwr_frame = sub.reset_index(drop=True)
print(f"\ngwr_frame shape: {gwr_frame.shape}")
print(f"coords: {coords.shape}, y: {y.shape}, X: {X.shape}")
print(f"Any NaN in frame? {gwr_frame[['PctBach','PctFB','PctBlack','PctRural','X','Y']].isna().any().any()}")

# --- block 9 ---

import inspect
import mgwr
from mgwr.gwr import GWR, GWRResults, GWRResultsLite
from mgwr import sel_bw
import mgwr.gwr as gwr_mod

print("GWR signature:")
print(inspect.signature(GWR.__init__))
print()
print("get_AICc source:")
print(inspect.getsource(sel_bw.get_AICc))
print()
print("GWRResultsLite attrs:", [a for a in dir(GWRResultsLite) if not a.startswith('_')])
print()
print("GWRResults attrs:", [a for a in dir(GWRResults) if not a.startswith('_')])

# --- block 10 (wrapped in try/except) ---
try:

    import inspect
    import mgwr.sel_bw as sel_bw
    print("sel_bw contents:", [a for a in dir(sel_bw) if not a.startswith('_')])
    print()
    gwr_src = inspect.getsource(type(sel_bw.GWR))
    print("sel_bw.GWR methods:", [a for a in dir(sel_bw.GWR) if not a.startswith('_')])
    print()
    # fit method of mgwr GWR
    print(inspect.getsource(sel_bw.GWR.fit))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 11 ---

import inspect
import mgwr.sel_bw as sel_bw
print(inspect.getsource(sel_bw.get_CV))
print("="*60)
print("sel_bw.GWR methods:", [a for a in dir(sel_bw.GWR) if not a.startswith('_')])

# --- block 12 (wrapped in try/except) ---
try:

    import time, numpy as np
    from mgwr.gwr import GWR

    t0 = time.time()
    m_test = GWR(coords, y, X, bw=150000.0, kernel='bisquare', fixed=True, constant=True)
    t1 = time.time()
    print("fit time: %.2f s" % (t1 - t0))
    print("n:", m_test.n)
    print("has influ:", hasattr(m_test, 'influ'), type(getattr(m_test, 'influ', None)))
    print("has resid_response:", hasattr(m_test, 'resid_response'))
    print("has predy:", hasattr(m_test, 'predy'))
    print("has aicc:", hasattr(m_test, 'aicc'), getattr(m_test, 'aicc', None))
    print("has tr_S:", hasattr(m_test, 'tr_S'), getattr(m_test, 'tr_S', None))
    print("has llf:", hasattr(m_test, 'llf'))
    print("R2:", m_test.R2)
    print()
    from mgwr import sel_bw
    cv = sel_bw.get_CV(m_test)
    print("CV (mean squared):", cv, " CV RMSE:", np.sqrt(cv))
    print("AICc via sel_bw.get_AICc:", sel_bw.get_AICc(m_test))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 13 (wrapped in try/except) ---
try:

    import time, numpy as np
    from mgwr import sel_bw

    t0 = time.time()
    m_test.fit()
    t1 = time.time()
    print("fit time: %.2f s" % (t1 - t0))
    print("n:", m_test.n)
    print("has influ:", hasattr(m_test, 'influ'), type(getattr(m_test, 'influ', None)))
    print("has resid_response:", hasattr(m_test, 'resid_response'))
    print("has predy:", hasattr(m_test, 'predy'))
    print("has aicc:", m_test.aicc)
    print("has tr_S:", m_test.tr_S)
    print("R2:", m_test.R2, " adj_R2:", m_test.adj_R2)
    cv = sel_bw.get_CV(m_test)
    print("CV (mean squared):", cv, " CV RMSE:", np.sqrt(cv))
    print("AICc via sel_bw.get_AICc:", sel_bw.get_AICc(m_test))
    print("params shape:", m_test.params.shape)
    print("predy shape:", m_test.predy.shape, "range:", m_test.predy.min(), m_test.predy.max())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 14 ---

import numpy as np
print("coords:", coords.shape, coords.dtype)
print("y:", y.shape, y.dtype)
print("X:", X.shape, X.dtype)
print("X row 0:", X[0])
print()
import inspect
from mgwr.gwr import GWR
src = inspect.getsource(GWR.fit)
print(src)

# --- block 15 ---

import inspect
from mgwr.gwr import GWR
print(inspect.getsource(GWR._local_fit))

# --- block 16 ---

import inspect
from mgwr.gwr import GWR
print(inspect.getsource(GWR.__init__))

# --- block 17 (wrapped in try/except) ---
try:

    import inspect
    from mgwr.glm import GLM
    print(inspect.getsource(GLM.__init__))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 18 ---

import inspect
import mgwr.gwr as gwr_mod
src = inspect.getsource(gwr_mod)
# print import lines
for line in src.splitlines():
    if line.startswith('import') or line.startswith('from'):
        print(line)

# --- block 19 ---

import inspect
import numpy as np
from spglm.glm import GLM
print(inspect.getsource(GLM.__init__))
print("="*60)
import mgwr.gwr as g
print("GLM module file:", inspect.getfile(GLM))

# --- block 20 ---

import inspect
from spglm.iwls import _compute_betas_gwr
print(inspect.getsource(_compute_betas_gwr))

# --- block 21 ---

import numpy as np, time
from mgwr.gwr import GWR

y2d = y.reshape(-1, 1)
X2d = X  # (159,3), constant added inside
print("y2d:", y2d.shape, "X2d:", X2d.shape)

t0 = time.time()
m2 = GWR(coords, y2d, X2d, bw=150000.0, kernel='bisquare', fixed=True, constant=True, n_jobs=1)
r = m2.fit()
print("fit time: %.2f s" % (time.time() - t0))
print("predy:", r.predy.shape, "range:", r.predy.min(), r.predy.max())
print("params:", r.params.shape)
print("R2:", r.R2, "adjR2:", r.adj_R2)
print("aicc:", r.aicc)
from mgwr import sel_bw
print("get_AICc:", sel_bw.get_AICc(r))
print("CV RMSE:", np.sqrt(sel_bw.get_CV(r)))
print("resid_response:", np.asarray(r.resid_response).shape)
print("influ:", np.asarray(r.influ).shape)

# --- block 22 (wrapped in try/except) ---
try:

    import numpy as np
    from scipy.spatial.distance import cdist

    d = cdist(coords, coords)
    print("max pairwise distance: %.0f m" % d.max())
    print("state X extent: %.0f m, Y extent: %.0f m" % (coords[:,0].ptp(), coords[:,1].ptp()))
    diagonal = np.hypot(coords[:,0].ptp(), coords[:,1].ptp())
    print("bounding-box diagonal: %.0f m" % diagonal)
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 23 (wrapped in try/except) ---
try:

    import numpy as np, time
    import pandas as pd
    from mgwr.gwr import GWR
    from mgwr import sel_bw

    # 2-D y fixes the scalar-dot bug in mgwr._local_fit
    y2d = y.reshape(-1, 1)

    bandwidths = np.arange(10000, 600001, 10000)  # 10 km .. 600 km, step 10 km
    print("candidate bandwidths (%d): %d..%d m, step 10000 m" % (len(bandwidths), bandwidths.min(), bandwidths.max()))

    rows = []
    t0 = time.time()
    for bw in bandwidths:
        m = GWR(coords, y2d, X2d, bw=float(bw), kernel='bisquare', fixed=True, constant=True, n_jobs=1)
        r = m.fit()
        aicc = float(r.aicc)
        cvrmse = float(np.sqrt(sel_bw.get_CV(r)))
        rows.append({'bandwidth': bw, 'AICc': aicc, 'CV_RMSE': cvrmse})
    print("loop time: %.1f s" % (time.time() - t0))

    bandwidth_table = pd.DataFrame(rows)
    bandwidth_table.to_csv("bandwidth_table.csv", index=False)
    print(bandwidth_table.to_string(index=False))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 24 ---

import numpy as np
from pyproj import Transformer

print("coords head:\n", coords[:3])
print("gwr_frame X,Y head:\n", gwr_frame[['X','Y']].head(3).values)
print("coords matches X,Y:", bool(np.allclose(coords, gwr_frame[['X','Y']].values)))

# Which UTM zone do the stored X/Y actually correspond to?
tr14 = Transformer.from_crs("EPSG:4326", "EPSG:32614", always_xy=True)
tr15 = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
lat, lon = demo['Latitude'].values, demo['Longitud'].values
x14, y14 = tr14.transform(lon, lat)
x15, y15 = tr15.transform(lon, lat)
e14 = np.hypot(x14 - coords[:,0], y14 - coords[:,1]).max()
e15 = np.hypot(x15 - coords[:,0], y5 - coords[:,1]).max() if False else np.hypot(x15 - coords[:,0], y15 - coords[:,1]).max()
print("max deviation stored vs projected (m)  zone14: %.1f   zone15: %.1f" % (e14, e15))

# --- block 25 ---

import numpy as np
from pyproj import Transformer

print("counties.crs:", counties.crs.to_wkt() if counties.crs is not None else None)
print("merged_gdf.crs:", merged_gdf.crs.to_wkt() if merged_gdf.crs is not None else None)

for zone in (13, 14, 15, 16):
    tr = Transformer.from_crs("EPSG:4326", "EPSG:326%d" % zone, always_xy=True)
    xx, yy = tr.transform(lon, lat)
    dev = np.hypot(xx - coords[:,0], yy - coords[:,1])
    print("zone %d: median dev %.1f m, p90 %.1f m, max %.1f m, n_dev>100m: %d" %
          (zone, np.median(dev), np.percentile(dev,90), dev.max(), (dev>100).sum()))

# --- block 26 ---

import numpy as np
from scipy.spatial.distance import cdist

# --- 1) Standardized predictors + intercept for the local WLS fits ---
mu  = X.mean(axis=0)
sd  = X.std(axis=0, ddof=0)
assert np.all(sd > 0), "a predictor is constant; cannot standardize"
Xs   = (X - mu) / sd
X_design = np.column_stack([np.ones(len(y)), Xs])   # (159, 4)

print("Standardization (population sd):")
for name, m, s in zip(['PctFB','PctBlack','PctRural'], mu, sd):
    print("  %-8s mean=%8.3f  sd=%7.3f" % (name, m, s))
print("X_design shape:", X_design.shape)
print("X_design column ranges (col0=intercept):")
print(np.round(X_design.min(axis=0),4), np.round(X_design.max(axis=0),4))
print("y range: %.4f .. %.4f" % (y.min(), y.max()))

# --- 2) Duplicate coordinate check ---
uniq = np.unique(np.round(coords, 3), axis=0)
n_unique_coords = int(len(uniq))
dup = len(y) - n_unique_coords
print("n_unique_coords:", n_unique_coords, "| duplicate coordinate pairs:", dup)
assert n_unique_coords == len(y), "duplicate locations would break GWR weighting"

# --- 3) Full 159x159 Euclidean distance matrix (UTM metres) ---
d = cdist(coords, coords, metric='euclidean')
assert d.shape == (159, 159) and np.allclose(np.diag(d), 0)
print("d shape:", d.shape, "min off-diagonal: %.1f m, max: %.1f m (state diagonal)"
      % (np.sort(d[np.triu_indices(159, k=1)])[0], d.max()))

# --- block 27 (wrapped in try/except) ---
try:

    import numpy as np

    # k-th nearest neighbour distance (k=12) for every county, from d
    d_sorted = np.sort(d, axis=1)              # (159,159), first column = 0 (self)
    d12      = d_sorted[:, 12]                 # 12th neighbour distance per county
    print("12th-nearest-neighbour distance per county: min %.0f m, max %.0f m" % (d12.min(), d12.max()))

    bw_lo = max(25_000.0, 1.02 * d12.max())    # strict: bisquare w>0 iff dist < bw
    bw_hi = 500_000.0
    bandwidth_grid = np.linspace(bw_lo, bw_hi, 48).round(0)
    print("grid bounds: %.0f m .. %.0f m, n candidates: %d, step ~ %.0f m"
          % (bandwidth_grid[0], bandwidth_grid[-1], len(bandwidth_grid),
             bandwidth_grid[1]-bandwidth_grid[0]))

    # verify the >=12 positively-weighted-obs constraint across the whole grid
    min_pos = np.minimum((bandwidth_grid[:, None] > d).sum(axis=1).min(), None)
    n_pos = (bandwidth_grid[:, None][None] > d[None]).sum(axis=2)   # (nbw, 159): counts per county
    n_pos_min_grid = n_pos.min(axis=1)
    print("min #positive-weight obs per county across grid: min %d (at bw=%.0f), max %d (at bw=%.0f)"
          % (n_pos_min_grid.min(), bandwidth_grid[n_pos_min_grid.argmin()],
             n_pos_min_grid.max(), bandwidth_grid[n_pos_min_grid.argmax()]))
    assert n_pos_min_grid.min() >= 12, "grid violates the 12-positively-weighted-obs rule"
    print("bandwidth_grid (m):", bandwidth_grid.tolist())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 28 (wrapped in try/except) ---
try:

    import numpy as np

    # count of positively-weighted (bisquare, dist < bw) observations, per county, per candidate bw
    n_pos = (d[None, :, :] < bandwidth_grid[:, None, None]).sum(axis=(1, 2))  # (48,) -> per-county min
    n_pos_min_grid = n_pos.min(axis=1)
    print("per-county min #positive-weight obs across grid: min %d (bw=%.0f m), max %d (bw=%.0f m)"
          % (n_pos_min_grid.min(), bandwidth_grid[int(n_pos_min_grid.argmin())],
             n_pos_min_grid.max(), bandwidth_grid[int(n_pos_min_grid.argmax())]))
    assert n_pos_min_grid.min() >= 12, "grid violates the 12-positively-weighted-obs rule"
    print("OK: every county has >= 12 positively weighted obs for every candidate bandwidth")
    print("bandwidth_grid (m):", bandwidth_grid.tolist())
    print("n candidates:", len(bandwidth_grid))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 29 ---

import numpy as np
# (48, 159): number of counties j with dist(i,j) < bw for each candidate bw and county i
n_pos_bw = (d[None, :, :] < bandwidth_grid[:, None, None]).sum(axis=2)
n_pos_min_grid = n_pos_bw.min(axis=1)   # min over counties, per bandwidth
print("per-county min #positive-weight obs, across the grid:")
print("  overall min %d at bw=%.0f m; overall max %d at bw=%.0f m"
      % (n_pos_min_grid.min(), bandwidth_grid[int(n_pos_min_grid.argmin())],
         n_pos_min_grid.max(), bandwidth_grid[int(n_pos_min_grid.argmax())]))
assert n_pos_min_grid.min() >= 12
print("OK: every county has >= 12 positively weighted (bisquare) observations for every candidate bandwidth")
print("bandwidth_grid: %d candidates, %.0f m .. %.0f m" % (len(bandwidth_grid), bandwidth_grid[0], bandwidth_grid[-1]))

# --- block 30 ---

import numpy as np
print("STEP 4 FINAL CHECK")
print("X_design     :", type(X_design).__name__, X_design.shape, "intercept col all 1:", bool((X_design[:,0]==1).all()))
print("y            :", type(y).__name__, y.shape, "finite:", bool(np.isfinite(y).all()))
print("d            :", type(d).__name__, d.shape, "symmetric:", bool(np.allclose(d, d.T)), "diag~0:", bool(np.max(np.abs(np.diag(d)))<1e-9))
print("bandwidth_grid:", type(bandwidth_grid).__name__, bandwidth_grid.shape, "all >0:", bool((bandwidth_grid>0).all()), "min %.0f m, max %.0f m" % (bandwidth_grid.min(), bandwidth_grid.max()))
print("n_unique_coords:", n_unique_coords, "(== 159:", n_unique_coords==159, ")")
print("coords (unchanged source):", coords.shape, "X (raw predictors):", X.shape)
print("Note: coords are in UTM zone 16N metres (verified against lat/lon); zone-15 projection was shown to deviate ~564 km, so stored coords were used as-is per the step spec.")

# --- block 31 ---
import numpy as np

n = len(y)
Xd = X_design
print("Xd shape:", Xd.shape, "| col means:", np.round(Xd.mean(0), 3), "| col stds:", np.round(Xd.std(0), 3))
print("bandwidth_grid: n=%d, min=%.0f m, max=%.0f m" % (len(bandwidth_grid), bandwidth_grid.min(), bandwidth_grid.max()))

def bisquare_weights(D, h):
    t = D / h
    W = np.zeros_like(D)
    m = t < 1.0
    W[m] = (1.0 - t[m]**2)**2
    return W

def solve_local(A, b):
    # A: (m,p,p) symmetric PSD per local fit; b: (m,p). SVD-style solve with ridge + eigenvalue floor.
    lam = 1e-8 * (np.trace(A[0]) / A.shape[-1])
    A = A + lam * np.eye(A.shape[-1])
    evals, Evecs = np.linalg.eigh(A)
    s2 = np.maximum(evals, 1e-12 * evals.max(axis=1, keepdims=True))
    c = np.einsum('jkl,jl->jk', Evecs, b)
    beta = np.einsum('jkl,jk,jl->jl', Evecs, 1.0/s2, c)
    Ainv = np.einsum('jkl,jk,jlm->jlm', Evecs, 1.0/s2, Evecs)
    return beta, Ainv

def local_AB(W, Xtr, ytr):
    A = np.einsum('ji,ik,il->jkl', W, Xtr, Xtr)
    b = np.einsum('ji,ik,i->jk', W, Xtr, ytr)
    return A, b

def in_sample(h):
    W = bisquare_weights(d, h)
    A, b = local_AB(W, Xd, y)
    beta, Ainv = solve_local(A, b)
    pred = np.einsum('ij,ij->i', Xd, beta)
    rss = float(np.sum((y - pred)**2))
    Hdiag = np.einsum('ij,ijk,ik->i', Xd, Ainv, Xd)
    trace_h = float(Hdiag.sum())
    sw = W.sum(1); sw2 = (W*W).sum(1)
    min_ess = float(np.min(sw*sw/sw2))
    return pred, rss, trace_h, min_ess

# --- sanity check: huge bandwidth (all W=1) must reproduce global OLS ---
pred_h, rss_h, tr_h, ess_h = in_sample(2_000_000.0)
beta_ols, *_ = np.linalg.lstsq(Xd, y, rcond=None)
pred_ols = Xd @ beta_ols
print("check@2000km  max|pred_igwr - pred_ols| = %.3e | rss=%.4f | trace_h=%.3f | min_ess=%.3f" %
      (np.max(np.abs(pred_h - pred_ols)), rss_h, tr_h, ess_h))

for h in (bandwidth_grid[0], bandwidth_grid[len(bandwidth_grid)//2], bandwidth_grid[-1]):
    p, r, t, e = in_sample(float(h))
    print("h=%.0f m  rss=%.1f  trace_h=%.2f  min_ess=%.1f  pred range=[%.2f, %.2f]" %
          (h, r, t, e, p.min(), p.max()))

# --- block 32 ---
import numpy as np
# Isolate solve_local: single global fit vs np.linalg.solve
A0 = Xd.T @ Xd
b0 = Xd.T @ y
A0b, b0b = A0[None,:].copy(), b0[None,:]
beta0, Ainv0 = solve_local(A0b, b0b)
beta_ref = np.linalg.solve(A0, b0)
print("beta  eig-decomp :", np.round(beta0[0], 5))
print("beta  np.solve   :", np.round(beta_ref, 5))
print("Ainv  eig vs inv : max diff = %.3e" % np.max(np.abs(Ainv0[0] - np.linalg.inv(A0))))
print("eigs of A0:", np.linalg.eigvalsh(A0))
# batched eigh vs per-matrix eigh
evals_b, Evecs_b = np.linalg.eigh(A0b)
print("batched evals:", evals_b)
print("Evecs orthonormal check (should be 0):",
      np.max(np.abs(Evecs_b[0] @ Evecs_b[0].T - np.eye(4))))
# manual reconstruction
c0 = Evecs_b[0].T @ b0
print("manual Ainv*b  :", np.round(Evecs_b[0] @ (Evecs_b[0] * 1.0/evals_b[0]) @ b0, 4))
# check d sanity
print("d max = %.1f m, d diag max = %.3e, symmetric: %s" % (d.max(), np.abs(np.diag(d)).max(), np.allclose(d, d.T)))

# --- block 33 ---
import numpy as np
Q = Evecs_b[0]          # (4,4), A0 = Q diag(s) Q.T
s = evals_b[0]
# corrected formula
c_correct = Q.T @ b0                     # (4,)
beta_fixed = Q @ (c_correct / s)
Ainv_fixed = (Q * (1.0/s)).T @ Q.T       # Q diag(1/s) Q.T
print("beta_fixed vs np.solve : max diff = %.3e" % np.max(np.abs(beta_fixed - np.linalg.solve(A0, b0))))
print("Ainv_fixed @ A0 - I    : max diff = %.3e" % np.max(np.abs(Ainv_fixed @ A0 - np.eye(4))))
# batched einsum versions
Qb, sb = Evecs_b, evals_b
c_b = np.einsum('jlk,jl->jk', Qb, b0b)
beta_b = np.einsum('jkl,jk,jl->jl', Qb, 1.0/sb, c_b)
Ainv_b = np.einsum('jkl,jk,jkm->jlm', Qb, 1.0/sb, Qb)
print("batched beta vs np.solve : max diff = %.3e" % np.max(np.abs(beta_b[0] - np.linalg.solve(A0, b0))))
print("batched Ainv vs true inv : max diff = %.3e" % np.max(np.abs(Ainv_b[0] - np.linalg.inv(A0))))

# --- block 34 (wrapped in try/except) ---
try:
    import numpy as np

    # ---------- corrected, unambiguous vectorized WLS via eigen-decomposition ----------
    def solve_local(A, b):
        """A: (m,p,p) symmetric per local fit, b: (m,p). Returns beta (m,p), Ainv (m,p,p)."""
        ridge = 1e-8 * (np.trace(A[0]) / A.shape[-1]) * np.eye(A.shape[-1])
        A = A + ridge
        s, Q = np.linalg.eigh(A)                 # A = Q diag(s) Q.T  (eigh contract)
        D = 1.0 / np.maximum(s, 1e-12 * s.max(axis=1, keepdims=True))
        c    = Q.transpose(0, 2, 1) @ b[:, :, None]          # (m,p,1) = Q.T b
        beta = (Q @ (c * D)).squeeze(-1)                     # (m,p)   = Q D (Q.T b)
        Ainv = (Q * D[:, :, None]).transpose(0, 2, 1) @ Q    # (m,p,p) = Q D Q.T
        return beta, Ainv

    # ---------- cross-check vs naive per-point np.linalg.solve ----------
    def naive_gwr(band):
        W = bisquare_weights(d, band)
        pred = np.empty(n); Hdiag = np.empty(n)
        for i in range(n):
            w = W[i]; sel = w > 0
            Xs, ys, ws = Xd[sel], y[sel], w[sel]
            Aw = (Xs * ws[:, None]).T @ Xs + 1e-10 * np.eye(4)
            bw = (Xs * ws[:, None]).T @ ys
            beta = np.linalg.solve(Aw, bw)
            pred[i] = Xd[i] @ beta
            Hdiag[i] = Xd[i] @ np.linalg.solve(Aw, Xd[i])
        return pred, Hdiag

    for band in (109515.0, 308912.0, 2000000.0):
        W = bisquare_weights(d, band)
        A, b = local_AB(W, Xd, y)
        beta, Ainv = solve_local(A, b)
        pred_v = np.einsum('ij,ij->i', Xd, beta)
        H_v    = np.einsum('ij,ijk,ik->i', Xd, Ainv, Xd)
        pred_n, H_n = naive_gwr(band)
        print("band %.0f m | d(beta)=%.3e | d(Hdiag)=%.3e | trace_v=%.4f | trace_n=%.4f | rss_v=%.2f"
              % (band, np.abs(pred_v - pred_n).max(), np.abs(H_v - H_n).max(), H_v.sum(), H_n.sum(),
                 np.sum((y - pred_v)**2)))
    # global limit must give trace = p = 4
    pred_g, H_g = naive_gwr(2000000.0)
    print("global-limit trace (should be 4): %.6f | beta vs OLS: %.3e" %
          (H_g.sum(), np.abs(Xd @ np.linalg.solve(Xd.T@Xd, y).squeeze() - pred_g).max()))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 35 ---
import numpy as np

def solve_local(A, b):
    """A: (m,p,p) symmetric per local fit, b: (m,p). Returns beta (m,p), Ainv (m,p,p)."""
    ridge = 1e-8 * (np.trace(A[0]) / A.shape[-1]) * np.eye(A.shape[-1])
    A = A + ridge
    s, Q = np.linalg.eigh(A)
    D = 1.0 / np.maximum(s, 1e-12 * s.max(axis=1, keepdims=True))
    c    = Q.transpose(0, 2, 1) @ b[:, :, None]          # (m,p,1)
    beta = (Q @ (c * D[:, :, None])).squeeze(-1)         # (m,p)
    Ainv = (Q * D[:, :, None]).transpose(0, 2, 1) @ Q    # (m,p,p)
    return beta, Ainv

for band in (109515.0, 308912.0, 2000000.0):
    W = bisquare_weights(d, band)
    A, b = local_AB(W, Xd, y)
    beta, Ainv = solve_local(A, b)
    pred_v = np.einsum('ij,ij->i', Xd, beta)
    H_v    = np.einsum('ij,ijk,ik->i', Xd, Ainv, Xd)
    pred_n, H_n = naive_gwr(band)
    print("band %.0f m | d(pred)=%.3e | d(Hdiag)=%.3e | trace_v=%.4f | trace_n=%.4f | rss_v=%.2f"
          % (band, np.abs(pred_v - pred_n).max(), np.abs(H_v - H_n).max(), H_v.sum(), H_n.sum(),
             np.sum((y - pred_v)**2)))
pred_g, H_g = naive_gwr(2000000.0)
beta_ols = np.linalg.solve(Xd.T @ Xd, Xd.T @ y)
print("global-limit trace (should be 4): %.6f | d(pred global, OLS) = %.3e" %
      (H_g.sum(), np.abs(pred_g - Xd @ beta_ols).max()))
print("OLS global: R2=%.4f" % (1 - np.sum((y - Xd@beta_ols)**2)/np.sum((y-y.mean())**2)))

# --- block 36 (wrapped in try/except) ---
try:
    import numpy as np

    def solve_local(A, b):
        """A: (m,p,p) symmetric PSD per local fit, b: (m,p). Returns beta (m,p), Ainv (m,p,p)."""
        ridge = 1e-8 * (np.trace(A[0]) / A.shape[-1]) * np.eye(A.shape[-1])
        A = A + ridge
        s, Q = np.linalg.eigh(A)                                   # A = Q diag(s) Q.T
        D = 1.0 / np.maximum(s, 1e-12 * s.max(axis=1, keepdims=True))
        c    = Q.transpose(0, 2, 1) @ b[:, :, None]                # (m,p,1) = Q.T b
        beta = (Q @ (c * D[:, :, None])).squeeze(-1)               # (m,p)   = Q D (Q.T b)
        Ainv = (Q * D[None, :, :]) @ Q.transpose(0, 2, 1)          # (m,p,p) = Q D Q.T
        return beta, Ainv

    # check Ainv is a true inverse on the band-109515 fits
    band = 109515.0
    W = bisquare_weights(d, band)
    A, b = local_AB(W, Xd, y)
    beta, Ainv = solve_local(A, b)
    resid = np.einsum('jkl,jlm->jkm', Ainv, A)
    print("max |Ainv @ A - I| = %.3e" % np.max(np.abs(resid - np.eye(4))))

    for band in (109515.0, 308912.0, 2000000.0):
        W = bisquare_weights(d, band)
        A, b = local_AB(W, Xd, y)
        beta, Ainv = solve_local(A, b)
        pred_v = np.einsum('ij,ij->i', Xd, beta)
        H_v    = np.einsum('ij,ijk,ik->i', Xd, Ainv, Xd)
        pred_n, H_n = naive_gwr(band)
        print("band %.0f m | d(pred)=%.3e | d(Hdiag)=%.3e | trace_v=%.4f | trace_n=%.4f | rss_v=%.2f"
              % (band, np.abs(pred_v - pred_n).max(), np.abs(H_v - H_n).max(), H_v.sum(), H_n.sum(),
                 np.sum((y - pred_v)**2)))
    print("GLOBAL check: trace at 2e6 m should be ~4 (weights ~1): see above")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 37 ---
import numpy as np

def solve_local(A, b):
    """A: (m,p,p) symmetric PSD per local fit, b: (m,p). Returns beta (m,p), Ainv (m,p,p).
    Uses eigen-decomposition (SVD-equivalent for symmetric matrices) with ridge + eigenvalue
    floor instead of direct inversion. A = Q diag(s) Q.T  =>  A^-1 = Q diag(D) Q.T."""
    ridge = 1e-8 * (np.trace(A[0]) / A.shape[-1]) * np.eye(A.shape[-1])
    A = A + ridge
    s, Q = np.linalg.eigh(A)                                   # A = Q diag(s) Q.T (cols of Q are evecs)
    D = 1.0 / np.maximum(s, 1e-12 * s.max(axis=1, keepdims=True))
    c    = Q.transpose(0, 2, 1) @ b[:, :, None]                # (m,p,1) = Q.T b
    beta = (Q @ (c * D[:, :, None])).squeeze(-1)               # (m,p)   = Q D (Q.T b)
    Ainv = (Q * D[:, :, None]) @ Q.transpose(0, 2, 1)          # (m,p,p) = Q diag(D) Q.T
    return beta, Ainv

# strict identity check + cross-validation vs naive per-county solver
for band in (109515.0, 308912.0, 500000.0):
    W = bisquare_weights(d, band)
    A, b = local_AB(W, Xd, y)
    beta, Ainv = solve_local(A, b)
    resid = np.max(np.abs(np.einsum('jkl,jlm->jkm', Ainv, A) - np.eye(4)))
    pred_v = np.einsum('ij,ij->i', Xd, beta)
    H_v    = np.einsum('ij,ijk,ik->i', Xd, Ainv, Xd)
    pred_n, H_n = naive_gwr(band)
    print("band %8.0f m | |Ainv@A-I|=%.2e | d(pred)=%.2e | d(Hdiag)=%.2e | trace_v=%.4f | trace_n=%.4f | rss=%.2f"
          % (band, resid, np.abs(pred_v - pred_n).max(), np.abs(H_v - H_n).max(), H_v.sum(), H_n.sum(),
             np.sum((y - pred_v)**2)))

# --- block 38 ---
import numpy as np
# single-matrix ground truth
A0 = Xd.T @ Xd
s, Q = np.linalg.eigh(A0)
D = 1.0 / s
print("A0 = Q diag(s) Q.T  check: max|A0 - Q@diag(s)@Q.T| = %.3e" %
      np.max(np.abs(A0 - (Q * D) @ Q.T * s)))  # (Q*D)@Q.T = A0
# correct inverse: Q diag(D) Q.T  -> scale COLUMNS (last axis) of Q
Ainv_row = (Q * D[:, None]) @ Q.T     # WRONG: diag(D) Q Q.T = diag(D)
Ainv_col = (Q * D[None, :]) @ Q.T     # RIGHT: Q diag(D) Q.T
print("row-scaling  |Ainv@A - I| = %.3e" % np.max(np.abs(Ainv_row @ A0 - np.eye(4))))
print("col-scaling  |Ainv@A - I| = %.3e" % np.max(np.abs(Ainv_col @ A0 - np.eye(4))))
print("col-scaling  vs np.linalg.inv = %.3e" % np.max(np.abs(Ainv_col - np.linalg.inv(A0))))

# --- block 39 (wrapped in try/except) ---
try:
    import numpy as np

    def solve_local(A, b):
        """A: (m,p,p) symmetric PSD per local fit, b: (m,p). Returns beta (m,p), Ainv (m,p,p).
        Eigen-decomposition (SVD-equivalent) with ridge + eigenvalue floor:
        A = Q diag(s) Q.T  =>  A^-1 = Q diag(D) Q.T  (scale COLUMNS of Q)."""
        ridge = 1e-8 * (np.trace(A[0]) / A.shape[-1]) * np.eye(A.shape[-1])
        A = A + ridge
        s, Q = np.linalg.eigh(A)
        D = 1.0 / np.maximum(s, 1e-12 * s.max(axis=1, keepdims=True))
        c    = Q.transpose(0, 2, 1) @ b[:, :, None]                # (m,p,1) = Q.T b
        beta = (Q @ (c * D[:, :, None])).squeeze(-1)               # (m,p)   = Q D (Q.T b)
        Ainv = (Q * D[None, :, :]) @ Q.transpose(0, 2, 1)          # (m,p,p) = Q diag(D) Q.T
        return beta, Ainv

    ok = True
    for band in (109515.0, 308912.0, 500000.0):
        W = bisquare_weights(d, band)
        A, b = local_AB(W, Xd, y)
        beta, Ainv = solve_local(A, b)
        resid = np.max(np.abs(np.einsum('jkl,jlm->jkm', Ainv, A) - np.eye(4)))
        pred_v = np.einsum('ij,ij->i', Xd, beta)
        H_v    = np.einsum('ij,ijk,ik->i', Xd, Ainv, Xd)
        pred_n, H_n = naive_gwr(band)
        dp, dh = np.abs(pred_v - pred_n).max(), np.abs(H_v - H_n).max()
        ok &= (resid < 1e-8) and (dp < 1e-4) and (dh < 1e-3)
        print("band %8.0f m | |Ainv@A-I|=%.2e | d(pred)=%.2e | d(Hdiag)=%.2e | trace_v=%.4f | trace_n=%.4f | rss=%.2f"
              % (band, resid, dp, dh, H_v.sum(), H_n.sum(), np.sum((y - pred_v)**2)))
    print("CROSS-VALIDATION PASSED" if ok else "CROSS-VALIDATION FAILED")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 40 (wrapped in try/except) ---
try:
    import numpy as np, pandas as pd

    Xd, yv, dmat = X_design, y, d
    n, p = len(yv), Xd.shape[1]
    print("n =", n, " p =", p, " grid:", len(bandwidth_grid), "km range:",
          round(bandwidth_grid.min()/1000,1), "-", round(bandwidth_grid.max()/1000,1))

    def bisquare_w(drow, h):
        w = np.zeros_like(drow)
        m = drow < h
        u = drow[m]/h
        w[m] = (1.0-u*u)**2
        return w

    def gwr_insample(dmat, Xd, yv, h):
        """Local WLS via SVD at every location (in-sample). Returns pred, rss, trace(H), min effective neighbors."""
        n = len(yv)
        pred = np.empty(n); hdiag = np.empty(n); eff = np.empty(n)
        for i in range(n):
            w = bisquare_w(dmat[i], h)
            sw = np.sqrt(w)
            M = sw[:,None]*Xd                      # (n,4)
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            eps = 1e-10*max(S[0], 1.0)
            small = S < eps
            Utr = U.T @ (sw*yv);  Utr[small] = 0.0
            beta = Vt.T @ (Utr/S)
            pred[i] = Xd[i] @ beta
            Utx = U.T @ Xd[i];  Utx[small] = 0.0
            hdiag[i] = np.sum((Utx/S)**2)          # H_ii of local WLS hat matrix
            eff[i] = w.sum()
        rss = float(np.sum((yv-pred)**2))
        return pred, rss, float(hdiag.sum()), float(eff.min())

    def aicc_gwr(n, p, rss, tr):
        if not np.isfinite(rss) or rss <= 0: return np.nan
        den = n - tr - 2*p - 1
        if den <= 0: return np.nan
        return float(n*np.log(rss/n) + n*(tr + 2*p + 1)/den)

    rows = []
    for h in bandwidth_grid:
        _, rss, tr, effmin = gwr_insample(dmat, Xd, yv, float(h))
        rows.append(dict(bandwidth=float(h), rss=rss, trace_h=tr,
                         aicc=aicc_gwr(n, p, rss, tr), min_eff_neighbors=effmin))
    bw_in = pd.DataFrame(rows)
    print(bw_in.head(6).to_string(index=False))
    print("...")
    print(bw_in.tail(4).to_string(index=False))
    print("finite AICc count:", np.isfinite(bw_in['aicc']).sum())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 41 (wrapped in try/except) ---
try:
    def gwr_insample(dmat, Xd, yv, h):
        """Local WLS via SVD at every location (in-sample). Returns pred, rss, trace(H), min effective neighbors."""
        n = len(yv)
        pred = np.empty(n); hdiag = np.empty(n); eff = np.empty(n)
        for i in range(n):
            w = bisquare_w(dmat[i], h)
            sw = np.sqrt(w)
            M = sw[:,None]*Xd
            U, S, Vt = np.linalg.svd(M, full_matrices=False)   # M(159,4) = U(159,k) diag(S) Vt(k,4), k=4
            eps = 1e-10*max(S[0], 1.0)
            small = S < eps
            Utr = U.T @ (sw*yv)      # (4,159)@(159,) = (4,)
            Utr[small] = 0.0
            beta = Vt.T @ (Utr/S)    # (4,4)@(4,) = (4,)
            pred[i] = Xd[i] @ beta
            Utx = Xd[i] @ U          # (4,)@(159,4) = (4,)
            Utx[small] = 0.0
            hdiag[i] = np.sum((Utx/S)**2)
            eff[i] = w.sum()
        rss = float(np.sum((yv-pred)**2))
        return pred, rss, float(hdiag.sum()), float(eff.min())

    rows = []
    for h in bandwidth_grid:
        _, rss, tr, effmin = gwr_insample(dmat, Xd, yv, float(h))
        rows.append(dict(bandwidth=float(h), rss=rss, trace_h=tr,
                         aicc=aicc_gwr(n, p, rss, tr), min_eff_neighbors=effmin))
    bw_in = pd.DataFrame(rows)
    print(bw_in.head(6).to_string(index=False))
    print("...")
    print(bw_in.tail(4).to_string(index=False))
    print("finite AICc count:", np.isfinite(bw_in['aicc']).sum())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 42 (wrapped in try/except) ---
try:
    import numpy as np, pandas as pd
    from sklearn.model_selection import KFold

    Xd, yv, dmat = X_design, y, d
    n, p = len(yv), Xd.shape[1]

    def gwr_insample(dmat, Xd, yv, h):
        nn = len(yv)
        pred = np.empty(nn); hdiag = np.empty(nn); eff = np.empty(nn)
        for i in range(nn):
            w = bisquare_w(dmat[i], h); sw = np.sqrt(w)
            M = sw[:,None]*Xd
            U, S, Vt = np.linalg.svd(M, full_matrices=False)   # U(159,4) S(4) Vt(4,4)
            eps = 1e-10*max(S[0],1.0); small = S < eps
            Utr = U.T @ (sw*yv); Utr[small]=0.0
            beta = Vt.T @ (Utr/S)
            pred[i] = Xd[i] @ beta
            hdiag[i] = np.sum(U[i,:]**2)      # (H=UU^T)_ii = ||U[i,:]||^2
            eff[i] = w.sum()
        return pred, float(np.sum((yv-pred)**2)), float(hdiag.sum()), float(eff.min())

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    def cv_rmse(h):
        errs = np.empty(n)
        for tr_idx, te_idx in kf.split(yv):
            D = dmat[te_idx, tr_idx]               # (te, tr)
            for k in range(len(te_idx)):
                w = bisquare_w(D[k], h); sw = np.sqrt(w)
                M = sw[:,None]*Xd[tr_idx]
                U,S,Vt = np.linalg.svd(M, full_matrices=False)
                eps=1e-10*max(S[0],1.0); small=S<eps
                Utr = U.T@(sw*yv[tr_idx]); Utr[small]=0.0
                beta = Vt.T@(Utr/S)
                errs[te_idx[k]] = yv[te_idx[k]] - Xd[te_idx[k]]@beta
        return float(np.sqrt(np.mean(errs**2)))

    rows = []
    for h in bandwidth_grid:
        _, rss, tr, effmin = gwr_insample(dmat, Xd, yv, float(h))
        rows.append(dict(bandwidth=float(h), rss=rss, trace_h=tr,
                         aicc=aicc_gwr(n, p, rss, tr), min_eff_neighbors=effmin,
                         cv_rmse=cv_rmse(float(h)), status="ok"))
    bandwidth_table = pd.DataFrame(rows)
    print(bandwidth_table.to_string(index=False,
         formatters={"bandwidth":"{:,.0f}".format,"aicc":"{:.1f}".format,
                     "cv_rmse":"{:.3f}".format,"trace_h":"{:.1f}".format,"rss":"{:.1f}".format}))

    fin = np.isfinite(bandwidth_table["aicc"])
    if fin.any():
        optimal_bandwidth = float(bandwidth_table.loc[bandwidth_table["aicc"].idxmin(),"bandwidth"])
        opt_row = bandwidth_table.loc[bandwidth_table["aicc"].idxmin()]
        sel_rule = "lowest finite AICc"
    else:
        optimal_bandwidth = float(bandwidth_table.loc[bandwidth_table["cv_rmse"].idxmin(),"bandwidth"])
        opt_row = bandwidth_table.loc[bandwidth_table["cv_rmse"].idxmin()]
        sel_rule = "lowest CV RMSE (AICc not finite)"
    optimal_aicc = float(opt_row["aicc"]); optimal_cv_rmse = float(opt_row["cv_rmse"])
    print("\nselection rule:", sel_rule)
    print("optimal_bandwidth = %.0f m (%.1f km) | AICc = %.2f | CV_RMSE = %.3f"
          % (optimal_bandwidth, optimal_bandwidth/1000,
             optimal_aicc if np.isfinite(optimal_aicc) else np.nan, optimal_cv_rmse))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 43 ---
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

Xd, yv, dmat = X_design, y, d
n, p = len(yv), Xd.shape[1]

def bisq(drow, h):
    w = np.zeros_like(drow); m = drow < h
    u = drow[m]/h; w[m] = (1.0-u*u)**2; return w

def local_beta(tr_idx, ti, h):
    w = bisq(dmat[ti, tr_idx], h); sw = np.sqrt(w)
    M = sw[:,None]*Xd[tr_idx]
    U,S,Vt = np.linalg.svd(M, full_matrices=False)
    nz = S >= 1e-10*max(S[0],1.0)
    Utr = U.T @ (sw*yv[tr_idx]); Utr[nz] = Utr[nz]/S[nz]
    return Vt.T @ Utr

def gwr_insample(h):
    pred = np.empty(n); hdiag = np.empty(n); eff = np.empty(n)
    for i in range(n):
        w = bisq(dmat[i], h); sw = np.sqrt(w)
        U,S,Vt = np.linalg.svd(sw[:,None]*Xd, full_matrices=False)
        nz = S >= 1e-10*max(S[0],1.0)
        Utr = U.T @ (sw*yv); Utr[nz] = Utr[nz]/S[nz]
        pred[i] = Xd[i] @ (Vt.T @ Utr)
        hdiag[i] = np.sum(U[i,nz]**2)
        eff[i] = w.sum()
    return pred, float(np.sum((yv-pred)**2)), float(hdiag.sum()), float(eff.min())

def aicc_fn(rss, tr):
    if not np.isfinite(rss) or rss <= 0: return np.nan
    den = n - tr - 2*p - 1
    return np.nan if den <= 0 else float(n*np.log(rss/n) + n*(tr + 2*p + 1)/den)

folds = list(KFold(5, shuffle=True, random_state=42).split(yv))
def cv_rmse(h):
    errs = np.empty(n)
    for tr_idx, te_idx in folds:
        for ti in te_idx:
            errs[ti] = yv[ti] - Xd[ti] @ local_beta(tr_idx, ti, h)
    return float(np.sqrt(np.mean(errs**2)))

rows = []
for h in bandwidth_grid:
    _, rss, tr, effmin = gwr_insample(float(h))
    rows.append(dict(bandwidth=float(h), rss=rss, trace_h=tr,
                     aicc=aicc_fn(rss, tr), min_eff_neighbors=effmin,
                     cv_rmse=cv_rmse(float(h)), status="ok"))
bandwidth_table = pd.DataFrame(rows)
print(bandwidth_table.to_string(index=False, formatters={"bandwidth":"{:,.0f}".format,
      "aicc":"{:.1f}".format,"cv_rmse":"{:.3f}".format,"trace_h":"{:.1f}".format,"rss":"{:.1f}".format}))

fin = np.isfinite(bandwidth_table["aicc"])
if fin.any():
    i_opt = int(bandwidth_table["aicc"].idxmin()); sel_rule = "lowest finite AICc"
else:
    i_opt = int(bandwidth_table["cv_rmse"].idxmin()); sel_rule = "lowest CV RMSE"
optimal_bandwidth = float(bandwidth_table.loc[i_opt, "bandwidth"])
optimal_aicc = float(bandwidth_table.loc[i_opt, "aicc"])
optimal_cv_rmse = float(bandwidth_table.loc[i_opt, "cv_rmse"])
print("\nselection rule:", sel_rule)
print("optimal_bandwidth = %.0f m (%.1f km) | AICc = %s | CV_RMSE = %.3f" %
      (optimal_bandwidth, optimal_bandwidth/1000,
       "finite %.2f" % optimal_aicc if np.isfinite(optimal_aicc) else "nan", optimal_cv_rmse))

# ---------- Step 6: final model + reference bandwidths ----------
def gwr_predict(h):
    pred = np.empty(n)
    for i in range(n):
        w = bisq(dmat[i], h); sw = np.sqrt(w)
        U,S,Vt = np.linalg.svd(sw[:,None]*Xd, full_matrices=False)
        nz = S >= 1e-10*max(S[0],1.0)
        Utr = U.T @ (sw*yv); Utr[nz] = Utr[nz]/S[nz]
        pred[i] = Xd[i] @ (Vt.T @ Utr)
    return pred

bw_low = float(bandwidth_grid[0]); bw_med = float(bandwidth_grid[len(bandwidth_grid)//2])
ref_bw = list(dict.fromkeys([bw_low, bw_med, optimal_bandwidth]))   # unique, keep order
ref_names = ["low %.0f km" % (bw/1000) for bw in ref_bw]

final_predictions = gwr_predict(optimal_bandwidth)
final_residuals = yv - final_predictions
sst = float(np.sum((yv - yv.mean())**2))
final_summary = dict(optimal_bandwidth_m=optimal_bandwidth,
                     R2=1 - float(np.sum(final_residuals**2))/sst,
                     RMSE=float(np.sqrt(np.mean(final_residuals**2))),
                     MAE=float(np.mean(np.abs(final_residuals))),
                     selection_rule=sel_rule,
                     AICc=optimal_aicc, CV_RMSE=optimal_cv_rmse)
reference_predictions = {nm: gwr_predict(bw) for nm, bw in zip(ref_names, ref_bw)}
prediction_diffs = pd.DataFrame({nm: final_predictions - reference_predictions[nm]
                                 for nm in ref_names})

print("\n=== FINAL MODEL (bandwidth = %.0f m) ===" % optimal_bandwidth)
print("R2 = %.4f | RMSE = %.4f | MAE = %.4f" % (final_summary["R2"], final_summary["RMSE"], final_summary["MAE"]))
print("final_predictions shape", final_predictions.shape,
      "| range [%.2f, %.2f] vs observed [%.2f, %.2f]" %
      (final_predictions.min(), final_predictions.max(), yv.min(), yv.max()))
print("\n=== REFERENCE BANDWIDTHS (prediction difference vs optimal) ===")
cmp = pd.DataFrame({"bandwidth_m": ref_bw,
                    "mean|diff|": [prediction_diffs[c].abs().mean() for c in ref_names],
                    "max|diff|": [prediction_diffs[c].abs().max() for c in ref_names],
                    "corr_with_opt": [np.corrcoef(reference_predictions[c], final_predictions)[0,1]
                                      for c in ref_names]}, index=ref_names)
print(cmp.to_string())

# ---------- Step 7: combined difference figure ----------
os.makedirs("pred_results", exist_ok=True)
km = bandwidth_table["bandwidth"].values/1000.0
dA = bandwidth_table["aicc"].values - (optimal_aicc if np.isfinite(optimal_aicc) else np.nan)
dCV = bandwidth_table["cv_rmse"].values - optimal_cv_rmse

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1.0]})
h_lines, l_labels = [], []
ln, = ax1.plot(km, dA, "o-", ms=3, lw=1.4, color="tab:blue", label="ΔAICc vs optimal")
h_lines += [ln]; l_labels += ["ΔAICc vs optimal"]
ax1b = ax1.twinx()
ln, = ax1b.plot(km, dCV, "s--", ms=3, lw=1.4, color="tab:orange", label="ΔCV RMSE vs optimal")
h_lines += [ln]; l_labels += ["ΔCV RMSE vs optimal (right axis)"]
ax1.axvline(optimal_bandwidth/1000, color="red", ls=":", lw=1.5,
            label="optimal = %.0f km" % (optimal_bandwidth/1000))
h_lines += [plt.Line2D([], [], color="red", ls=":", lw=1.5,
            label="optimal = %.0f km" % (optimal_bandwidth/1000))]
ax1.set_ylabel("ΔAICc vs optimal"); ax1b.set_ylabel("ΔCV RMSE vs optimal")
ax1.set_title("GWR bandwidth search - Georgia counties (target: PctBach)\n"
              "differences from the optimal fixed bandwidth (bisquare kernel, 5-fold CV)")
ax1.legend(handles=h_lines, labels=l_labels, loc="upper left", fontsize=9)

ln, = ax2.plot(km, bandwidth_table["aicc"], color="tab:blue", alpha=0.35, lw=1, label="AICc")
ax2b = ax2.twinx()
ln2, = ax2b.plot(km, bandwidth_table["cv_rmse"], color="tab:orange", alpha=0.35, lw=1, label="CV RMSE")
ax2.axvline(optimal_bandwidth/1000, color="red", ls=":", lw=1.5)
ax2.set_ylabel("AICc (faint)"); ax2b.set_ylabel("CV RMSE (faint)")
ax2.set_xlabel("Bandwidth (km)")
ax2.set_title("Absolute model metrics (faint overlay)", fontsize=9)
ax2.legend(loc="upper left", fontsize=8)
ax2b.legend(loc="upper right", fontsize=8)

plot_path = "pred_results/GA_GWR.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight"); plt.close(fig)
print("\nfigure written:", plot_path, os.path.getsize(plot_path), "bytes")
