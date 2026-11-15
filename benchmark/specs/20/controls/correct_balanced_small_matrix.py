# Correct: balanced training set = the occurrence cells + as many background cells > 2 km away
# (~400 rows, all 7 layers), background drawn from an integer np.argwhere index array. The feature
# matrix is small and the only big 2-D array is the (N, 2) index array.
import geopandas as gpd, rasterio, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score
pts = gpd.read_file('dataset/sn_w_minoccs.gpkg')
names = ['tasgrav_IR', 'tasgrav_IR_1VD', 'tasmag_TMI', 'tasmag_TMI_1VD', 'tasrad_K_pct', 'tasrad_Th_ppm', 'tasrad_U_ppm']
arrs = []
for n in names:
    with rasterio.open(f'dataset/{n}.tif') as s:
        a = s.read(1).astype('float32'); a[a == s.nodata] = np.nan; arrs.append(a); tr = s.transform
X = np.stack(arrs, -1)
valid = ~np.isnan(X).any(-1)
r, c = rasterio.transform.rowcol(tr, pts.geometry.x.values, pts.geometry.y.values)
r = np.array(r); c = np.array(c)
rng = np.random.default_rng(42)
occ = np.zeros(valid.shape, bool); occ[r, c] = True
dist = ndimage.distance_transform_edt(~occ) * 50
cand = np.argwhere(valid & (dist > 2000))
pos = np.unique(np.c_[r, c], axis=0)
pos = pos[valid[pos[:, 0], pos[:, 1]]]
neg = cand[rng.choice(len(cand), len(pos), replace=False)]
Xd = np.vstack([X[pos[:, 0], pos[:, 1]], X[neg[:, 0], neg[:, 1]]]); y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
Xtr, Xte, ytr, yte = train_test_split(Xd, y, test_size=0.3, stratify=y, random_state=1)
rf = RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=4).fit(Xtr, ytr)
print('test AUC', roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]))
print('CV AUC', cross_val_score(RandomForestClassifier(300, random_state=1, n_jobs=4), Xd, y, cv=5, scoring='roc_auc').mean())
rf_final = RandomForestClassifier(n_estimators=300, random_state=1, n_jobs=4).fit(Xd, y)
prob = np.full(valid.shape, np.nan, 'float32')
prob[valid] = rf_final.predict_proba(X[valid])[:, 1]
fig, ax = plt.subplots(figsize=(9, 10))
im = ax.imshow(prob, cmap='viridis', vmin=0, vmax=1); plt.colorbar(im, ax=ax, label='Prospectivity probability')
ax.set_title('Random forest Sn-W prospectivity')
plt.savefig('pred_results/mineral_prospectivity.png', dpi=100); plt.close()
