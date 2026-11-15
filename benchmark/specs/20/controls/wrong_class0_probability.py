# Wrong: maps column 0 of predict_proba, i.e. P(no deposit), instead of the deposit probability.
import glob, numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
arrs = []
for f in sorted(glob.glob('dataset/*.tif')):
    with rasterio.open(f) as s:
        a = s.read(1).astype(float); a[a == s.nodata] = np.nan; arrs.append(a); T = s.transform
cube = np.stack(arrs); mask = np.isfinite(cube).all(0)
occ = gpd.read_file('dataset/sn_w_minoccs.gpkg')
y = rasterize([(g, 1) for g in occ.buffer(1000)], out_shape=mask.shape, transform=T)
X = cube[:, mask].T; yy = y[mask]
Xb, yb = RandomUnderSampler(random_state=0).fit_resample(X, yy)
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0, n_jobs=4).fit(Xb, yb)
prob = np.full(mask.shape, np.nan); prob[mask] = rf.predict_proba(X)[:, 0]     # column 0 = P(no deposit)
plt.imshow(prob, vmin=0, vmax=1); plt.colorbar(); plt.savefig('pred_results/mineral_prospectivity.png')
