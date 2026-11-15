# Wrong: adds easting/northing as features -> the forest memorises deposit locations (spatial leakage, map = location blobs).
import glob, os, numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
files = sorted(glob.glob('dataset/*.tif')); arrs = []
for f in files:
    with rasterio.open(f) as s:
        a = s.read(1).astype('float32'); arrs.append(np.where(a == s.nodata, np.nan, a)); T = s.transform; shp = s.shape
r, c = np.indices(shp)
arrs += [c.astype('float32'), r.astype('float32')]
cube = np.stack(arrs); mask = np.isfinite(cube).all(0)
occ = gpd.read_file('dataset/sn_w_minoccs.gpkg')
y = rasterize([(g, 1) for g in occ.geometry.buffer(1000)], out_shape=shp, transform=T, fill=0)
X = cube[:, mask].T; yv = y[mask]
Xb, yb = RandomUnderSampler(random_state=0).fit_resample(X, yv)
Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.33, random_state=0)
rf = RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=4).fit(Xtr, ytr)
auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]); print('AUC', auc)
prob = np.full(shp, np.nan, 'float32'); prob[mask] = rf.predict_proba(X)[:, 1]
del cube, r, c
plt.figure(figsize=(7, 9)); plt.imshow(prob, cmap='viridis', vmin=0, vmax=1); plt.colorbar(label='P(Sn-W)')
plt.title(f'AUC {auc:.3f}'); plt.savefig('pred_results/mineral_prospectivity.png')
