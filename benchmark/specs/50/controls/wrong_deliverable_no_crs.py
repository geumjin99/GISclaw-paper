"""Wrong: the reference analysis of task 50, with every geographic deliverable then written without a CRS;
the deliverables must be georeferenced.

Seagrass habitat suitability from kriged temperature/salinity and a presence-only MaxEnt-type model.

All layers are in EPSG:8857 (Equal Earth, metres).
  1. Pairwise dissolve of USA_seagrass (30 polygons -> 1 multipart feature).
  2. 5 000 random presence points inside the dissolved seagrass (seed 0).
  3-4. Kriging of `temp` and `salinity` from the 33 196 Global_ocean_measurements points onto a
     25 km grid. Empirical Bayesian Kriging is ArcGIS-only; the open substitute is ordinary
     kriging with a linear semivariogram (the model type named in the original EBK call) fitted
     to the empirical variogram (lags <= 1 000 km, 4 000-point subsample) and a local search
     neighbourhood of the 50 nearest points (EBK: max_local_points=50).
     Cells are kept where they intersect Global_shallow_waters (the prediction mask).
  5. Presence-only prediction (MaxEnt) implemented as infinitely-weighted logistic regression
     (Fithian & Hastie 2013): presence points (spatially thinned at 2 km, inside the study area
     USA_shallow_waters) vs background = all 25 km cells of the study area, background weight
     100; features linear, quadratic, product and hinge (10 knots per variable) of the two kriged
     rasters; L2 regularisation; cloglog output; presence cut-off 0.5.
     Spatial vs random 3-fold CV AUC is printed to expose spatial-autocorrelation optimism.
Outputs: pred_results/trainfeatures2.shp (training points with explanatory values and fitted
probability), pred_results/seagrass_predict2.tif (global probability of presence),
pred_results/EBK_temp.tif, pred_results/EBK_salinity.tif, pred_results/seagrass_prediction.png
"""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from shapely.geometry import Point
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
CELL = 25000.0

ocean = gpd.read_file("dataset/Global_ocean_measurements.shp")
global_shallow = gpd.read_file("dataset/Global_shallow_waters.shp")
seagrass = gpd.read_file("dataset/USA_seagrass.shp")
usa_shallow = gpd.read_file("dataset/USA_shallow_waters.shp")
CRS = ocean.crs

# 1 dissolve
seagrass_dissolved = seagrass[["geometry"]].dissolve()
sg_geom = seagrass_dissolved.geometry.iloc[0]

# 2 random points in the dissolved seagrass
minx, miny, maxx, maxy = sg_geom.bounds
pts = []
from shapely.prepared import prep
psg = prep(sg_geom)
while len(pts) < 5000:
    cx = rng.uniform(minx, maxx, 20000); cy = rng.uniform(miny, maxy, 20000)
    for x_, y_ in zip(cx, cy):
        if psg.contains(Point(x_, y_)):
            pts.append((x_, y_))
            if len(pts) == 5000:
                break
usa_train = gpd.GeoDataFrame({"CID": np.arange(1, 5001)}, geometry=gpd.points_from_xy(*np.array(pts).T), crs=CRS)
print("random presence points:", len(usa_train))

# 3-4 kriging on a 25 km grid
gx0, gy0, gx1, gy1 = global_shallow.total_bounds
ncol = int(np.ceil((gx1 - gx0) / CELL)); nrow = int(np.ceil((gy1 - gy0) / CELL))
transform = from_origin(gx0, gy1, CELL, CELL)
shallow_mask = ~geometry_mask(global_shallow.geometry, (nrow, ncol), transform, all_touched=True)
study_mask = ~geometry_mask(usa_shallow.geometry, (nrow, ncol), transform, all_touched=True)
pred_mask = shallow_mask | study_mask
rows, cols = np.nonzero(pred_mask)
cx = gx0 + (cols + 0.5) * CELL; cy = gy1 - (rows + 0.5) * CELL
print("grid", (nrow, ncol), "cells to predict:", len(rows))

oxy = np.c_[ocean.geometry.x, ocean.geometry.y]
tree = cKDTree(oxy)


def fit_linear_variogram(z, n_sub=4000, max_lag=1.0e6, n_bins=20):
    idx = rng.choice(len(z), n_sub, replace=False)
    p, v = oxy[idx], z[idx]
    pairs = cKDTree(p).query_pairs(max_lag, output_type="ndarray")
    h = np.linalg.norm(p[pairs[:, 0]] - p[pairs[:, 1]], axis=1)
    g = 0.5 * (v[pairs[:, 0]] - v[pairs[:, 1]]) ** 2
    bins = np.linspace(0, max_lag, n_bins + 1)
    k = np.digitize(h, bins) - 1
    hm = np.array([h[k == i].mean() for i in range(n_bins) if (k == i).sum() > 30])
    gm = np.array([g[k == i].mean() for i in range(n_bins) if (k == i).sum() > 30])
    slope, nugget = np.polyfit(hm, gm, 1)
    return max(slope, 1e-12), max(nugget, 0.0)


def local_ok(z, slope, nugget, k=50, batch=4000):
    """Ordinary kriging, linear semivariogram, 50 nearest neighbours."""
    def gam(h):
        return np.where(h > 0, nugget + slope * h, 0.0)
    out = np.empty(len(cx)); var = np.empty(len(cx))
    for s in range(0, len(cx), batch):
        tx = np.c_[cx[s:s + batch], cy[s:s + batch]]
        d, nb = tree.query(tx, k=k)
        P = oxy[nb]                                             # (b, k, 2)
        D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
        b = len(tx)
        A = np.ones((b, k + 1, k + 1)); A[:, :k, :k] = gam(D); A[:, k, k] = 0.0
        rhs = np.ones((b, k + 1)); rhs[:, :k] = gam(d)
        w = np.linalg.solve(A, rhs[..., None])[..., 0]
        out[s:s + batch] = (w[:, :k] * z[nb]).sum(1)
        var[s:s + batch] = (w * rhs).sum(1)
    return out, var


ebk = {}
for var_name in ("temp", "salinity"):
    z = ocean[var_name].to_numpy(float)
    slope, nugget = fit_linear_variogram(z)
    est, kvar = local_ok(z, slope, nugget)
    grid = np.full((nrow, ncol), np.nan, dtype="float32"); grid[rows, cols] = est
    ebk[var_name] = grid
    print(f"{var_name}: linear variogram slope {slope:.3e}/m nugget {nugget:.3f}; surface {np.nanmin(grid):.2f}..{np.nanmax(grid):.2f}")
EBK_temp, EBK_salinity = ebk["temp"], ebk["salinity"]
prof = dict(driver="GTiff", height=nrow, width=ncol, count=1, dtype="float32", crs=CRS, transform=transform,
            nodata=np.nan, compress="lzw")
for nm, arr in (("EBK_temp", EBK_temp), ("EBK_salinity", EBK_salinity)):
    with rasterio.open(f"pred_results/{nm}.tif", "w", **prof) as dst:
        dst.write(arr, 1)


def sample(grid, x, y):
    c = ((np.asarray(x) - gx0) / CELL).astype(int); r = ((gy1 - np.asarray(y)) / CELL).astype(int)
    ok = (r >= 0) & (r < nrow) & (c >= 0) & (c < ncol)
    v = np.full(len(c), np.nan); v[ok] = grid[r[ok], c[ok]]
    return v


# 5 presence-only prediction (MaxEnt as infinitely-weighted logistic regression)
study_geom = usa_shallow.geometry.union_all()
pres = usa_train[usa_train.within(study_geom)].copy()
keep, taken = [], cKDTree(np.c_[pres.geometry.x, pres.geometry.y])
pxy = np.c_[pres.geometry.x, pres.geometry.y]
removed = np.zeros(len(pres), bool)
for i in range(len(pres)):                                      # spatial thinning, 2 km
    if removed[i]:
        continue
    keep.append(i)
    for j in taken.query_ball_point(pxy[i], 2000.0):
        if j != i:
            removed[j] = True
pres = pres.iloc[keep].copy()
br, bc = np.nonzero(study_mask)
bg = gpd.GeoDataFrame(geometry=gpd.points_from_xy(gx0 + (bc + 0.5) * CELL, gy1 - (br + 0.5) * CELL), crs=CRS)
pres["TRAINED_ID"] = 1; bg["TRAINED_ID"] = 0
train = gpd.GeoDataFrame(np.r_[0:0], crs=CRS) if False else gpd.GeoDataFrame(
    {"CLASS": np.r_[np.ones(len(pres), int), np.zeros(len(bg), int)]},
    geometry=list(pres.geometry) + list(bg.geometry), crs=CRS)
train["TEMP"] = sample(EBK_temp, train.geometry.x, train.geometry.y)
train["SALINITY"] = sample(EBK_salinity, train.geometry.x, train.geometry.y)
train = train.dropna(subset=["TEMP", "SALINITY"]).reset_index(drop=True)
print("training: presence", int(train.CLASS.sum()), "background", int((train.CLASS == 0).sum()))

mu = train[["TEMP", "SALINITY"]].mean().to_numpy(); sd = train[["TEMP", "SALINITY"]].std().to_numpy()
knots = [np.quantile(train[c], np.linspace(0.05, 0.95, 10)) for c in ("TEMP", "SALINITY")]
lo_hi = [(train[c].min(), train[c].max()) for c in ("TEMP", "SALINITY")]


def features(t, s):
    X = np.c_[t, s]
    Z = (X - mu) / sd
    F = [Z, Z ** 2, (Z[:, 0] * Z[:, 1])[:, None]]
    for j in range(2):
        lo, hi = lo_hi[j]
        for kk in knots[j]:
            F.append((np.clip(X[:, j] - kk, 0, None) / max(hi - kk, 1e-9))[:, None])     # forward hinge
            F.append((np.clip(kk - X[:, j], 0, None) / max(kk - lo, 1e-9))[:, None])     # reverse hinge
    return np.hstack(F)


Xtr = features(train.TEMP.to_numpy(), train.SALINITY.to_numpy()); ytr = train.CLASS.to_numpy()
wts = np.where(ytr == 1, 1.0, 100.0)
model = LogisticRegression(C=1.0, max_iter=5000)
model.fit(Xtr, ytr, sample_weight=wts)


def cloglog(X):
    eta = model.decision_function(X)
    eta = eta - np.log(np.average(np.exp(model.decision_function(Xtr[ytr == 0]))))   # normalise over background
    return 1 - np.exp(-np.exp(eta))


train["PROBABILITY"] = cloglog(Xtr)
train["PREDICTED"] = (train["PROBABILITY"] >= 0.5).astype(int)
train.to_file("pred_results/trainfeatures2.shp")

# model assessment: random vs spatial-block 3-fold CV (spatial autocorrelation / leakage check)
def cv_auc(groups):
    aucs = []
    for g in np.unique(groups):
        te = groups == g
        m = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr[~te], ytr[~te], sample_weight=wts[~te])
        if len(np.unique(ytr[te])) == 2:
            aucs.append(roc_auc_score(ytr[te], m.decision_function(Xtr[te])))
    return float(np.mean(aucs))


rand_groups = rng.integers(0, 3, len(train))
bx = np.floor(train.geometry.x / 500000).astype(int); by = np.floor(train.geometry.y / 500000).astype(int)
blk = (bx * 1000 + by).to_numpy()
ub = np.unique(blk); perm = rng.permutation(len(ub)); spatial_groups = perm[np.searchsorted(ub, blk)] % 3
auc_random, auc_spatial = cv_auc(rand_groups), cv_auc(spatial_groups)
print(f"3-fold CV AUC: random {auc_random:.3f}, spatial blocks (500 km) {auc_spatial:.3f}")

# global prediction over the shallow-water mask
grid_t, grid_s = EBK_temp[rows, cols], EBK_salinity[rows, cols]
in_shallow = shallow_mask[rows, cols]
prob = np.full((nrow, ncol), np.nan, dtype="float32")
prob[rows[in_shallow], cols[in_shallow]] = cloglog(features(grid_t[in_shallow], grid_s[in_shallow]))
seagrass_predict = prob
with rasterio.open("pred_results/seagrass_predict2.tif", "w", **prof) as dst:
    dst.write(seagrass_predict, 1)
print("predicted cells:", int(np.isfinite(prob).sum()), "share >= 0.5:", round(float(np.nanmean(prob >= 0.5)), 3))

fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(np.ma.masked_invalid(prob), extent=[gx0, gx0 + ncol * CELL, gy1 - nrow * CELL, gy1], cmap="viridis",
               vmin=0, vmax=1)
seagrass_dissolved.boundary.plot(ax=ax, color="red", linewidth=0.5)
plt.colorbar(im, ax=ax, shrink=0.6, label="probability of presence (cloglog)")
ax.set_title("Predicted seagrass habitat (MaxEnt-type presence-only model on kriged temperature & salinity)")
plt.savefig("pred_results/seagrass_prediction.png", dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL: the analysis above is the reference solution, but every geographic
# deliverable file is then written without a CRS. Expected: wrong.
def _write_without_crs(path):
    """Rewrite one file in place so that it carries no CRS. Returns a short description.
    .shp: drop the .prj sidecar. .tif/.tiff: rewrite without CRS (and drop .aux.xml, which can
    hold one). .gpkg: rewrite every spatial layer with crs=None. .geojson/.json: write projected
    coordinates (UTM if the data are lon/lat) without a 'crs' member - a lon/lat GeoJSON
    without 'crs' is WGS 84 by definition (RFC 7946), so dropping the member alone would not
    remove the CRS."""
    import os
    low = path.lower()
    stem, _ = os.path.splitext(path)
    if low.endswith(".shp"):
        d, b = os.path.split(stem)
        gone = []
        for n in os.listdir(d or "."):
            if n.lower() == (b + ".prj").lower():
                os.remove(os.path.join(d, n))
                gone.append(n)
        return f"removed {gone}"
    if low.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as r:
            prof, data = r.profile.copy(), r.read()
            tags = r.tags()
        prof.pop("crs", None)
        tmp = stem + ".__nocrs__" + os.path.splitext(path)[1]
        with rasterio.open(tmp, "w", **prof) as w:
            w.write(data)
            if tags:
                w.update_tags(**tags)
        os.replace(tmp, path)
        if os.path.exists(path + ".aux.xml"):
            os.remove(path + ".aux.xml")
        return "raster rewritten without CRS"
    if low.endswith(".gpkg"):
        import geopandas as gpd
        import pyogrio
        tmp = stem + ".__nocrs__.gpkg"
        for lyr, gt in pyogrio.list_layers(path):
            g = gpd.read_file(path, layer=lyr)
            if gt is not None:
                g = g.set_crs(None, allow_override=True)
            g.to_file(tmp, layer=lyr, driver="GPKG")
        os.replace(tmp, path)
        return "gpkg rewritten without CRS"
    if low.endswith((".geojson", ".json")):
        import geopandas as gpd
        g = gpd.read_file(path)
        if g.crs is None or g.crs.is_geographic:
            if g.crs is None:
                g = g.set_crs(4326)
            g = g.to_crs(g.estimate_utm_crs())
        g = g.set_crs(None, allow_override=True)
        os.remove(path)
        g.to_file(path, driver="GeoJSON")
        return "GeoJSON rewritten in projected coordinates without 'crs'"
    return "not a geographic file"


def _strip_deliverable_crs(patterns):
    import fnmatch
    import os
    for n in sorted(os.listdir("pred_results")):
        low = n.lower()
        if low.endswith(('.shp', '.geojson', '.gpkg', '.json', '.tif', '.tiff')) and any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns):
            if low.endswith(".json"):
                try:
                    import json
                    if json.load(open(os.path.join("pred_results", n))).get("type") is None:
                        continue
                except Exception:
                    continue
            print("no-CRS control:", n, _write_without_crs(os.path.join("pred_results", n)))


_strip_deliverable_crs(['*.geojson', '*.gpkg', '*.img', '*.json', '*.shp', '*.tif', '*.tiff'])
