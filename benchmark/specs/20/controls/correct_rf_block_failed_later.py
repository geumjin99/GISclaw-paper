# Correct: the RandomForestClassifier is constructed through getattr, so the code text has no
# "RandomForestClassifier(" and the only evidence of the method is the fitted forest in the namespace.
# This stands for a session in which the step that builds and fits the forest raises further down,
# while later steps predict with the fitted object.
import glob, numpy as np, rasterio, geopandas as gpd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.ensemble as ens
from rasterio.features import rasterize
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
arrs = []
for f in sorted(glob.glob('dataset/*.tif')):
    with rasterio.open(f) as s:
        a = s.read(1).astype('float32'); arrs.append(np.where(a == s.nodata, np.nan, a)); T = s.transform; shp = s.shape
cube = np.stack(arrs); mask = np.isfinite(cube).all(0)
occ = gpd.read_file('dataset/sn_w_minoccs.gpkg')
y = rasterize([(g, 1) for g in occ.geometry.buffer(1000)], out_shape=shp, transform=T, fill=0)
X = cube[:, mask].T; yv = y[mask]
rng = np.random.default_rng(3)
pi = np.flatnonzero(yv == 1); ni = rng.choice(np.flatnonzero(yv == 0), len(pi), replace=False)
flat = X[np.r_[pi, ni]]; yb = np.r_[np.ones(len(pi)), np.zeros(len(ni))]
Xtr, Xte, ytr, yte = train_test_split(flat, yb, test_size=0.3, random_state=0, stratify=yb)
rf = getattr(ens, 'RandomForest' + 'Classifier')(n_estimators=150, max_depth=10, random_state=0, n_jobs=4).fit(Xtr, ytr)
auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]); print('random-split test AUC', round(auc, 4))
prob = np.full(shp, np.nan, 'float32'); prob[mask] = rf.predict_proba(X)[:, 1]
del cube
plt.figure(figsize=(8, 10)); plt.imshow(prob, cmap='viridis', vmin=0, vmax=1); plt.colorbar(label='P(Sn-W)')
plt.savefig('pred_results/mineral_prospectivity.png', dpi=100); plt.close()
