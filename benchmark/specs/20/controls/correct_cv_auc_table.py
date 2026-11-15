# Correct: the AUCs come from a CV helper returning a tuple (mean, std, pooled OOF AUC), printed
# without the word AUC on the line, and are kept in a metrics table with a 'mean_fold_AUC' column
# (also written to CSV). No *auc* scalar, no AUC in the figure.
import geopandas as gpd, rasterio, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
names = ['tasgrav_IR', 'tasgrav_IR_1VD', 'tasmag_TMI', 'tasmag_TMI_1VD', 'tasrad_K_pct', 'tasrad_Th_ppm', 'tasrad_U_ppm']
L = []
for n in names:
    with rasterio.open(f'dataset/{n}.tif') as r:
        a = r.read(1).astype('float32'); a[a == r.nodata] = np.nan; L.append(a); tr = r.transform; shp = r.shape
cube = np.stack(L, -1); valid = ~np.isnan(cube).any(-1)
occ = gpd.read_file('dataset/sn_w_minoccs.gpkg')
lab = rasterize([(g, 1) for g in occ.buffer(1000).geometry], out_shape=shp, transform=tr, fill=0).astype(bool)
rng = np.random.default_rng(1)
pi = np.flatnonzero((lab & valid).ravel()); ni = np.flatnonzero((~lab & valid).ravel())
ni = rng.choice(ni, len(pi), replace=False)
Xs = cube.reshape(-1, 7)[np.r_[pi, ni]]; ys = np.r_[np.ones(len(pi)), np.zeros(len(ni))]
def cv_scores(splitter):
    res = []; oof = np.zeros(len(ys))
    for a, b in splitter.split(Xs, ys):
        m = RandomForestClassifier(100, max_depth=8, random_state=0, n_jobs=4).fit(Xs[a], ys[a])
        p = m.predict_proba(Xs[b])[:, 1]; oof[b] = p; res.append(roc_auc_score(ys[b], p))
    return np.mean(res), np.std(res), roc_auc_score(ys, oof)
m_str = cv_scores(StratifiedKFold(5, shuffle=True, random_state=1))
print('stratified 5-fold (undersampled):', m_str)
metrics = pd.DataFrame([{'validation': 'StratifiedKFold(5)', 'mean_fold_AUC': m_str[0], 'std': m_str[1], 'pooled_OOF_AUC': m_str[2]}])
print(metrics.to_string())
metrics.to_csv('pred_results/rf_cv_metrics.csv', index=False)
final = RandomForestClassifier(100, max_depth=8, random_state=0, n_jobs=4).fit(Xs, ys)
prob = np.full(shp, np.nan, 'float32'); prob[valid] = final.predict_proba(cube[valid])[:, 1]
del cube
plt.figure(figsize=(8, 10)); plt.imshow(prob, cmap='viridis', vmin=0, vmax=1); plt.colorbar(label='P(Sn-W)')
plt.title('Random forest prospectivity: tin-tungsten, Tasmania')
plt.savefig('pred_results/mineral_prospectivity.png', dpi=100); plt.close()
