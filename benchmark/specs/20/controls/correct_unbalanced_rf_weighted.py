# Correct: class_weight instead of undersampling, pixel subsample, deeper trees.
import glob, os, numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
arrs = []
for f in sorted(glob.glob('dataset/*.tif')):
    with rasterio.open(f) as s:
        a = s.read(1); arrs.append(np.where(a == s.nodata, np.nan, a)); T = s.transform; shp = s.shape
cube = np.stack(arrs)
mask = np.isfinite(cube).all(0)
occ = gpd.read_file('dataset/sn_w_minoccs.gpkg')
y = rasterize([(g, 1) for g in occ.geometry.buffer(1000)], out_shape=shp, transform=T)
X = cube[:, mask].T; yv = y[mask]
rng = np.random.default_rng(7); i = rng.choice(len(yv), 300000, replace=False)
Xtr, Xte, ytr, yte = train_test_split(X[i], yv[i], test_size=0.3, random_state=7, stratify=yv[i])
clf = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=7, n_jobs=4).fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]); print('AUC', auc)
out = np.full(shp, np.nan); out[mask] = clf.predict_proba(X)[:, 1]
plt.imshow(out, cmap='magma'); plt.colorbar(); plt.title(f'Prospectivity AUC={auc:.3f}')
plt.savefig('pred_results/mineral_prospectivity.png')
