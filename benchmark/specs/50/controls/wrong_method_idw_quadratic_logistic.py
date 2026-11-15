# Wrong: the task prescribes empirical Bayesian kriging; this interpolates temperature/salinity by IDW (the
# MaxEnt-like presence/background logistic model itself is acceptable).
import os, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
os.makedirs("pred_results", exist_ok=True)
oc = gpd.read_file("dataset/Global_ocean_measurements.shp"); gs = gpd.read_file("dataset/Global_shallow_waters.shp")
sg = gpd.read_file("dataset/USA_seagrass.shp"); us = gpd.read_file("dataset/USA_shallow_waters.shp")
pres = gpd.GeoDataFrame(geometry=sg.dissolve().sample_points(5000, rng=11).explode(index_parts=False).values, crs=sg.crs)
bg = gpd.GeoDataFrame(geometry=us.dissolve().sample_points(5000, rng=12).explode(index_parts=False).values, crs=sg.crs)
tree = cKDTree(np.c_[oc.geometry.x, oc.geometry.y])
def interp(x, y, col):
    d, i = tree.query(np.c_[x, y], k=12); w = 1 / np.maximum(d, 1) ** 2
    return (w * oc[col].values[i]).sum(1) / w.sum(1)
train = gpd.GeoDataFrame({"presence": np.r_[np.ones(len(pres)), np.zeros(len(bg))].astype(int)},
                         geometry=list(pres.geometry) + list(bg.geometry), crs=sg.crs)
train["temp"] = interp(train.geometry.x, train.geometry.y, "temp")
train["salinity"] = interp(train.geometry.x, train.geometry.y, "salinity")
train.to_file("pred_results/train_features.shp")
res = 50000; x0, y0, x1, y1 = gs.total_bounds
nc, nr = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
T = from_origin(x0, y1, res, res)
m = ~geometry_mask(gs.geometry, (nr, nc), T, all_touched=True)
r, c = np.nonzero(m); cx, cy = x0 + (c + .5) * res, y1 - (r + .5) * res
grid = pd.DataFrame({"temp": interp(cx, cy, "temp"), "salinity": interp(cx, cy, "salinity")})
# MaxEnt-like presence/background model: quadratic + product features, down-weighted background (IWLR-style)
mdl = make_pipeline(StandardScaler(), PolynomialFeatures(2), LogisticRegression(max_iter=2000, C=1.0))
wts = np.where(train.presence == 1, 1.0, 100.0)
mdl.fit(train[["temp", "salinity"]], train.presence, logisticregression__sample_weight=wts)
pred = mdl.predict_proba(grid[["temp", "salinity"]])[:, 1]
P = np.full((nr, nc), np.nan, "float32"); P[r, c] = pred
with rasterio.open("pred_results/seagrass_prediction.tif", "w", driver="GTiff", width=nc, height=nr, count=1,
                   dtype="float32", crs=sg.crs, transform=T, nodata=np.nan) as d:
    d.write(P, 1)
