# Correct: same RF pipeline, probability expressed in percent (0-100) on the map.
import glob, os, numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
def main():
    arrs = []
    for f in sorted(glob.glob('dataset/*.tif')):
        with rasterio.open(f) as s:
            a = s.read(1).astype('float32'); arrs.append(np.where(a == s.nodata, np.nan, a)); T = s.transform; shp = s.shape
    cube = np.stack(arrs); mask = np.isfinite(cube).all(0)
    occ = gpd.read_file('dataset/sn_w_minoccs.gpkg')
    y = rasterize([(g, 1) for g in occ.geometry.buffer(1000)], out_shape=shp, transform=T, fill=0)
    X = cube[:, mask].T; yv = y[mask]
    Xb, yb = RandomUnderSampler(random_state=1).fit_resample(X, yv)
    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.33, random_state=1, stratify=yb)
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=1, n_jobs=4).fit(Xtr, ytr)
    auc = roc_auc_score(yte, rf.predict_proba(Xte)[:, 1]); print('AUC', auc)
    pct = np.full(shp, np.nan, 'float32'); pct[mask] = 100 * rf.predict_proba(X)[:, 1]
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(pct, cmap='inferno', vmin=0, vmax=100); fig.colorbar(im, label='Sn-W prospectivity (%)')
    ax.set_title(f'RF prospectivity, test AUC {auc:.3f}')
    fig.savefig('pred_results/mineral_prospectivity.png')
main()
