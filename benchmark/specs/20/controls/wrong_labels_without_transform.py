# Wrong: labels rasterised in pixel space without the geotransform, so the deposits are misplaced on the grid.
import glob, numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
arrs = []
for f in sorted(glob.glob('dataset/*.tif')):
    with rasterio.open(f) as s:
        a = s.read(1).astype(float); a[a == s.nodata] = np.nan; arrs.append(a)
cube = np.stack(arrs); mask = np.isfinite(cube).all(0)
occ = gpd.read_file('dataset/sn_w_minoccs.gpkg')
# labels in pixel space without the geotransform: points scaled to the grid by bounding box (wrong placement/flip)
x0, y0, x1, y1 = occ.total_bounds
r = ((occ.geometry.y - y0) / (y1 - y0) * (mask.shape[0] - 1)).astype(int)
c = ((occ.geometry.x - x0) / (x1 - x0) * (mask.shape[1] - 1)).astype(int)
y = np.zeros(mask.shape, int)
for rr, cc in zip(r, c):
    y[max(rr - 20, 0):rr + 20, max(cc - 20, 0):cc + 20] = 1
X = cube[:, mask].T; yy = y[mask]
Xb, yb = RandomUnderSampler(random_state=0).fit_resample(X, yy)
rf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0, n_jobs=4).fit(Xb, yb)
prob = np.full(mask.shape, np.nan); prob[mask] = rf.predict_proba(X)[:, 1]
plt.imshow(prob, vmin=0, vmax=1); plt.colorbar(); plt.savefig('pred_results/mineral_prospectivity.png')
