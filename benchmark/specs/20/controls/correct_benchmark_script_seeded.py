# Correct: random forest with a fixed seed, run at top level (os.listdir order, last-layer nodata mask).
# Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, geopandas as gpd, rasterio
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
geotiffs = [os.path.join('dataset/', x) for x in os.listdir('dataset/') if '.tif' in x]
df = gpd.read_file('dataset/sn_w_minoccs.gpkg')
data = []
for fn in geotiffs:
    with rasterio.open(fn) as src:
        transform = src.transform; d = src.read(1); nodata_mask = d == src.nodata; d[nodata_mask] = np.nan; data.append(d)
data = np.stack(data)
labels = rasterize(((g, 1) for g in df.buffer(1000).geometry), out_shape=data[0].shape, fill=0, transform=transform).astype('float32')
labels[nodata_mask] = np.nan
X_pix = data.reshape((data.shape[0], -1)).T; y_pix = labels.flatten()
X = X_pix[~np.isnan(y_pix)]; y = y_pix[~np.isnan(y_pix)]
X_strat, y_strat = RandomUnderSampler(random_state=42).fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_strat, y_strat, test_size=0.33, random_state=42)
model = RandomForestClassifier(max_depth=3, n_jobs=1, random_state=42).fit(X_train, y_train)
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1]); roc_auc = auc(fpr, tpr)
pred_ar = np.zeros(nodata_mask.size, dtype='float32')
pred_ar[~nodata_mask.flatten()] = model.predict_proba(X_pix[~nodata_mask.flatten()])[:, 1]
pred_ar = pred_ar.reshape(nodata_mask.shape); pred_ar[nodata_mask] = np.nan
plt.imshow(pred_ar, vmin=0, vmax=1); plt.title('Probability Map'); plt.colorbar()
plt.savefig("pred_results/mineral_prospectivity.png")
