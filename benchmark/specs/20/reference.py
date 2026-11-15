"""Reference solution for task 20: random-forest prospectivity for tin-tungsten deposits, NE Tasmania.

Evidence: 7 co-registered 50 m rasters (gravity, magnetics + 1VD, K/Th/U radiometrics).
Labels: cells within 1 km of any of the 199 Sn-W occurrences = 1, all other valid cells = 0.
The rasters carry a LOCAL_CS "GDA94 / MGA zone 55" without EPSG code; their extent matches the
occurrences' EPSG:28355 coordinates, so no reprojection is needed. NoData (-99999) cells in any
layer are excluded. Classes are balanced by random undersampling; a random forest (fixed seeds)
is evaluated (i) with the task's random 67/33 split and (ii) with spatially blocked 5-fold CV
(10 km blocks), because buffered neighbouring pixels of the same deposit in train and test make
the random-split AUC optimistic (spatial leakage). The final model is fitted on all balanced
samples and mapped as P(Sn-W deposit) over every valid cell.
(The model factory is a private function, _make_rf: the capture harness snapshots the namespace
when a public top-level function returns, which is costly with the 4 M-cell arrays in scope.)
"""
import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler

SEED = 42
os.makedirs("pred_results", exist_ok=True)
tifs = sorted(glob.glob("dataset/*.tif"))
names = [os.path.basename(f)[:-4] for f in tifs]
layers = []
for f in tifs:
    with rasterio.open(f) as src:
        a = src.read(1).astype("float32")
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
        layers.append(a)
        transform, shape, bounds = src.transform, src.shape, src.bounds
stack = np.stack(layers)                                   # (7, 2633, 1876)
nodata_mask = np.isnan(stack).any(axis=0)

occ = gpd.read_file("dataset/sn_w_minoccs.gpkg")          # EPSG:28355, same grid space
labels = rasterize(((g, 1) for g in occ.buffer(1000).geometry), out_shape=shape,
                   transform=transform, fill=0, dtype="uint8").astype("float32")
labels[nodata_mask] = np.nan

rows, cols = np.indices(shape)
valid = ~nodata_mask
X_all = stack[:, valid].T
y_all = labels[valid]
xs, ys = rasterio.transform.xy(transform, rows[valid], cols[valid])
block = (np.floor(np.asarray(xs) / 10000).astype(int) * 1000 + np.floor(np.asarray(ys) / 10000).astype(int))
print("valid cells %d, positive %d (%.2f %%)" % (len(y_all), y_all.sum(), 100 * y_all.mean()))

rus = RandomUnderSampler(random_state=SEED)
X_bal, y_bal = rus.fit_resample(X_all, y_all)
g_bal = block[rus.sample_indices_]


def _make_rf():
    return RandomForestClassifier(n_estimators=200, max_depth=3, random_state=SEED, n_jobs=4)


# (i) random split, as specified by the task
X_tr, X_te, y_tr, y_te = train_test_split(X_bal, y_bal, test_size=0.33, random_state=SEED, stratify=y_bal)
rf = _make_rf().fit(X_tr, y_tr)
auc_random_split = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
# (ii) spatially blocked CV (10 km blocks) - leakage-aware estimate
oof = np.zeros(len(y_bal))
for tr, te in GroupKFold(n_splits=5).split(X_bal, y_bal, g_bal):
    oof[te] = _make_rf().fit(X_bal[tr], y_bal[tr]).predict_proba(X_bal[te])[:, 1]
auc_spatial_cv = roc_auc_score(y_bal, oof)
print("AUC random 67/33 split: %.3f   AUC spatial block CV: %.3f" % (auc_random_split, auc_spatial_cv))

final_model = _make_rf().fit(X_bal, y_bal)
prob_map = np.full(shape, np.nan, dtype="float32")
prob_map[valid] = final_model.predict_proba(X_all)[:, 1]
print("feature importances:", dict(zip(names, np.round(final_model.feature_importances_, 3))))

fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(prob_map, vmin=0, vmax=1, cmap="viridis",
               extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
occ.plot(ax=ax, color="red", markersize=4, label="Sn-W occurrences")
fig.colorbar(im, ax=ax, shrink=0.7, label="P(Sn-W, random forest)")
ax.set_title("Tin-tungsten prospectivity, NE Tasmania\nAUC %.3f (random split), %.3f (spatial CV)" % (auc_random_split, auc_spatial_cv))
ax.set_xlabel("Easting (m, MGA55)"); ax.set_ylabel("Northing (m)"); ax.legend(loc="lower left")
fig.savefig("pred_results/mineral_prospectivity.png", dpi=150, bbox_inches="tight")
