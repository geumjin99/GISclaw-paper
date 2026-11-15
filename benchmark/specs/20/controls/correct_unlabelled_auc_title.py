# Correct: the AUC is printed without a label - print(roc_auc_score(...), importances) - and shown in
# the map title 'RF, AUC=0.999'; no variable named *auc*. Positives: cells within 500 m of an
# occurrence; negatives: balanced random cells > 2 km away.
import geopandas as gpd, rasterio, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
g = gpd.read_file('dataset/sn_w_minoccs.gpkg')
names = ['tasgrav_IR', 'tasgrav_IR_1VD', 'tasmag_TMI', 'tasmag_TMI_1VD', 'tasrad_K_pct', 'tasrad_Th_ppm', 'tasrad_U_ppm']
L = []
for n in names:
    with rasterio.open(f'dataset/{n}.tif') as r:
        a = r.read(1).astype('float32'); a[a == r.nodata] = np.nan; L.append(a); tr = r.transform
X = np.stack(L, -1)
valid = ~np.isnan(X).any(-1)
rr, cc = rasterio.transform.rowcol(tr, g.geometry.x, g.geometry.y)
occ = np.zeros(valid.shape, bool); occ[np.array(rr), np.array(cc)] = True
d = ndimage.distance_transform_edt(~occ) * 50
pos = (d <= 500) & valid; far = (d > 2000) & valid
rng = np.random.default_rng(0)
P = np.argwhere(pos); N = np.argwhere(far); N = N[rng.choice(len(N), len(P), replace=False)]
Xf = np.vstack([X[P[:, 0], P[:, 1]], X[N[:, 0], N[:, 1]]]); y = np.r_[np.ones(len(P)), np.zeros(len(N))]
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.3, random_state=0, stratify=y)
rf = RandomForestClassifier(200, n_jobs=4, random_state=0, min_samples_leaf=5).fit(Xtr, ytr)
print(roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]), dict(zip(names, rf.feature_importances_.round(3))))
prob = np.full(valid.shape, np.nan, 'float32'); prob[valid] = rf.predict_proba(X[valid])[:, 1]
fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(prob, cmap='magma', vmin=0, vmax=1); plt.colorbar(im, label='Prospectivity probability', shrink=.7)
ax.set_title('Sn-W prospectivity, Tasmania (RF, AUC=%.3f)' % roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))
plt.savefig('pred_results/mineral_prospectivity.png', dpi=100); plt.close()
