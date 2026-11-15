# Correct: features as a named DataFrame, StandardScaler + RandomForest pipeline, undersampling by
# hand (numpy), 1 km buffered labels, probability mapped with imshow and the raster extent.
import glob, os, numpy as np, pandas as pd, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio.features import rasterize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
cols = {}
for f in sorted(glob.glob('dataset/*.tif')):
    with rasterio.open(f) as s:
        a = s.read(1).astype('float32'); a[a == s.nodata] = np.nan
        cols[os.path.basename(f)[:-4]] = a; T, shp, b = s.transform, s.shape, s.bounds
ok = np.all([np.isfinite(v) for v in cols.values()], axis=0)
feat = pd.DataFrame({k: v[ok] for k, v in cols.items()})
occ = gpd.read_file('dataset/sn_w_minoccs.gpkg')
lab = rasterize(((g, 1) for g in occ.buffer(1000)), out_shape=shp, transform=T, fill=0)[ok]
rng = np.random.default_rng(7)
pos = np.flatnonzero(lab == 1); neg = rng.choice(np.flatnonzero(lab == 0), len(pos), replace=False)
sel = np.r_[pos, neg]
Xtr, Xte, ytr, yte = train_test_split(feat.iloc[sel], lab[sel], test_size=0.33, random_state=7, stratify=lab[sel])
model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=150, max_depth=4, random_state=7, n_jobs=4))
model.fit(Xtr, ytr)
auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
prob = np.full(shp, np.nan, 'float32'); prob[ok] = model.predict_proba(feat)[:, 1]
fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(prob, vmin=0, vmax=1, cmap='viridis', extent=(b.left, b.right, b.bottom, b.top))
fig.colorbar(im, label='P(Sn-W)'); ax.set_title(f'RF prospectivity (AUC {auc:.3f})')
fig.savefig('pred_results/mineral_prospectivity.png')
