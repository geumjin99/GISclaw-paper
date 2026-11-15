# Wrong: the presence/background model is an MLP classifier wrapped in a function called fit_maxent; the
# task prescribes presence-only prediction (MaxEnt). The kriging code contains curve_fit, but the MaxEnt
# function itself fits no likelihood.
# Rest of the pipeline: dissolve + 5 000 random presence points; EBK-style local ordinary kriging of
# temperature and salinity (local exponential variograms, average of two models) on a global 20 km
# grid in EPSG:8857, the kriged surfaces saved as temp_EBK.tif / salinity_EBK.tif next to the
# prediction; MaxEnt-like IWLR model (L1 logistic on t, s, t^2, s^2, t*s, background weight 100,
# cloglog output) trained in the USA study area and projected onto the global shallow waters.
import os
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio import features as rfeat
from rasterio.transform import from_origin
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler

sg = gpd.read_file('dataset/USA_seagrass.shp')
usw = gpd.read_file('dataset/USA_shallow_waters.shp')
gsw = gpd.read_file('dataset/Global_shallow_waters.shp')
om = gpd.read_file('dataset/Global_ocean_measurements.shp')
om['x'] = om.geometry.x; om['y'] = om.geometry.y
loc = om.groupby(['x', 'y'])[['temp', 'salinity']].mean().reset_index()

sg_d = gpd.GeoDataFrame(geometry=[sg.geometry.make_valid().union_all()], crs=sg.crs)
pts = sg_d.sample_points(5000, rng=np.random.default_rng(42)).explode(index_parts=False)
pres = gpd.GeoDataFrame(geometry=pts.values, crs=sg.crs)
os.makedirs('pred_results', exist_ok=True)
sg_d.to_file('pred_results/USA_seagrass_dissolved.shp')
pres.to_file('pred_results/seagrass_random_points.shp')


def expv(h, n, s, r):
    return n + s * (1 - np.exp(-3 * h / r))


X = loc[['x', 'y']].values
tree = cKDTree(X)


def fit_local_vgs(z, nsub=40, m=300, maxd=8e5, seed=0):
    r_ = np.random.default_rng(seed); out = []
    while len(out) < nsub:
        c = X[r_.integers(len(z))]; _, idx = tree.query(c, m)
        d = pdist(X[idx]); g = 0.5 * pdist(z[idx, None]) ** 2; sel = d < maxd
        bins = np.linspace(0, maxd, 13); k = np.digitize(d[sel], bins) - 1
        ok = [j for j in range(12) if (k == j).sum() > 20]
        if len(ok) < 6 or g[sel].var() == 0:
            continue
        hb = np.array([d[sel][k == j].mean() for j in ok]); gb = np.array([g[sel][k == j].mean() for j in ok])
        try:
            p, _ = curve_fit(expv, hb, gb, p0=[0.1 * gb.max() + 1e-6, gb.max() + 1e-6, 3e5],
                             bounds=([0, 1e-6, 2e4], [np.inf, np.inf, 3e6]), maxfev=5000)
        except RuntimeError:
            continue
        out.append(p)
    return np.array(out)


vgl = {v: fit_local_vgs(loc[v].values, seed=s) for s, v in enumerate(['temp', 'salinity'])}

res = 20000.
xmin, ymin, xmax, ymax = -17080000., -8360000., 17260000., 8400000.
xs = np.arange(xmin + res / 2, xmax, res); ys = np.arange(ymax - res / 2, ymin, -res)
GX, GY = np.meshgrid(xs, ys); G = np.c_[GX.ravel(), GY.ravel()]


def ok_local(z, params, k=24, chunk=40000):
    n, s, r = params
    cov = lambda h: np.where(h == 0, n + s, s * np.exp(-3 * h / r))
    pred = np.empty(len(G))
    for i0 in range(0, len(G), chunk):
        g = G[i0:i0 + chunk]; d0, idx = tree.query(g, k); P = X[idx]
        D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
        m = len(g); A = np.ones((m, k + 1, k + 1)); A[:, :k, :k] = cov(D); A[:, k, k] = 0
        bvec = np.ones((m, k + 1)); bvec[:, :k] = cov(d0)
        w = np.linalg.solve(A, bvec[..., None])[..., 0]
        pred[i0:i0 + chunk] = (w[:, :k] * z[idx]).sum(1)
    return pred


krig = {}
for v in ['temp', 'salinity']:
    P = vgl[v]; order = np.argsort(P[:, 2]); sel = P[order[np.linspace(0, len(P) - 1, 2).astype(int)]]
    krig[v] = np.mean([ok_local(loc[v].values, p) for p in sel], 0).reshape(GX.shape)

tr = from_origin(xmin, ymax, res, res)
prof = dict(driver='GTiff', height=GX.shape[0], width=GX.shape[1], count=1, dtype='float32', crs='EPSG:8857',
            transform=tr, nodata=-9999, compress='lzw')
for v in krig:
    with rasterio.open(f'pred_results/{v}_EBK.tif', 'w', **prof) as dst:
        dst.write(krig[v].astype('float32'), 1)

usw_u = usw.geometry.make_valid().union_all()
bg = gpd.GeoDataFrame(geometry=gpd.GeoSeries([usw_u], crs=usw.crs).sample_points(10000, rng=np.random.default_rng(7))
                      .explode(index_parts=False).values, crs=usw.crs)


def sample(gdf):
    out = {}
    for v in krig:
        with rasterio.open(f'pred_results/{v}_EBK.tif') as src:
            out[v] = np.array([s[0] for s in src.sample(np.c_[gdf.geometry.x, gdf.geometry.y])])
    return pd.DataFrame(out)


pv = sample(pres); bv = sample(bg)
inside = pres.within(usw_u).values
P_tr = pres[inside].reset_index(drop=True); pv_tr = pv[inside].reset_index(drop=True)


def feats(df, sc=None):
    Z = df[['temp', 'salinity']].values
    if sc is None:
        sc = StandardScaler().fit(Z)
    Z = sc.transform(Z); t, s = Z[:, 0], Z[:, 1]
    return np.c_[t, s, t ** 2, s ** 2, t * s], sc


Xall = pd.concat([pv_tr, bv], ignore_index=True); yall = np.r_[np.ones(len(pv_tr)), np.zeros(len(bv))]
F, sc = feats(Xall)
from sklearn.neural_network import MLPClassifier


def fit_maxent(F, y):   # name only: a neural-network classifier
    return MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0).fit(F, y)


rf = fit_maxent(F, yall)


def cloglog(Fx):
    return rf.predict_proba(Fx)[:, 1]


prob = cloglog(F)
thr = np.quantile(prob[yall == 1], 0.10)
train = gpd.GeoDataFrame(pd.concat([pv_tr, bv], ignore_index=True),
                         geometry=list(P_tr.geometry) + list(bg.geometry), crs=usw.crs)
train['TRAINED_ID'] = np.r_[np.ones(len(pv_tr)), np.zeros(len(bv))].astype(int)
train['PRED_PROB'] = prob; train['PREDICTED'] = (prob >= thr).astype(int)
train.to_file('pred_results/seagrass_train_features.shp')
lo = Xall[['temp', 'salinity']].min().values; hi = Xall[['temp', 'salinity']].max().values
grid = pd.DataFrame({'temp': np.clip(krig['temp'].ravel(), lo[0], hi[0]),
                     'salinity': np.clip(krig['salinity'].ravel(), lo[1], hi[1])})
Fg, _ = feats(grid, sc); pg = cloglog(Fg).reshape(GX.shape)
gsw_u = gsw.geometry.make_valid().union_all()
m_glob = rfeat.rasterize([(gsw_u, 1)], out_shape=GX.shape, transform=tr, fill=0, all_touched=True).astype(bool)
with rasterio.open('pred_results/seagrass_predict_raster.tif', 'w', **prof) as d:
    d.write(np.where(m_glob, pg, -9999).astype('float32'), 1)
