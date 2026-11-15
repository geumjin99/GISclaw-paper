# Wrong: presence points drawn in the shallow-water study area instead of the dissolved seagrass polygons.
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, calculate_default_transform
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
oc = gpd.read_file('dataset/Global_ocean_measurements.shp'); gs = gpd.read_file('dataset/Global_shallow_waters.shp')
sg = gpd.read_file('dataset/USA_seagrass.shp'); us = gpd.read_file('dataset/USA_shallow_waters.shp')
diss = sg.dissolve()
pres = us.dissolve().sample_points(5000, rng=1).explode(index_parts=False)
pres = gpd.GeoDataFrame(geometry=pres.values, crs=sg.crs)
bg = us.dissolve().sample_points(5000, rng=2).explode(index_parts=False)
bg = gpd.GeoDataFrame(geometry=bg.values, crs=sg.crs)
xy = np.c_[oc.geometry.x, oc.geometry.y]; tree = cKDTree(xy)
def idw(x, y, col):
    d, i = tree.query(np.c_[x, y], k=12); w = 1 / np.maximum(d, 1) ** 2
    return (w * oc[col].values[i]).sum(1) / w.sum(1)
train = gpd.GeoDataFrame({'presence': np.r_[np.ones(len(pres)), np.zeros(len(bg))].astype(int)},
                         geometry=list(pres.geometry) + list(bg.geometry), crs=sg.crs)
train['temp'] = idw(train.geometry.x, train.geometry.y, 'temp')
train['salinity'] = idw(train.geometry.x, train.geometry.y, 'salinity')
rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=0).fit(train[['temp', 'salinity']], train.presence)
train.to_file('pred_results/train_features.geojson')
res = 50000; x0, y0, x1, y1 = gs.total_bounds
nc, nr = int((x1 - x0) / res) + 1, int((y1 - y0) / res) + 1
T = from_origin(x0, y1, res, res)
m = ~geometry_mask(gs.geometry, (nr, nc), T, all_touched=True)
r, c = np.nonzero(m); cx, cy = x0 + (c + .5) * res, y1 - (r + .5) * res
import pandas as pd
X = pd.DataFrame({'temp': idw(cx, cy, 'temp'), 'salinity': idw(cx, cy, 'salinity')})
P = np.full((nr, nc), np.nan, 'float32'); P[r, c] = rf.predict_proba(X)[:, 1]
dt, w, h = calculate_default_transform(sg.crs, 'EPSG:4326', nc, nr, x0, y1 - nr * res, x0 + nc * res, y1)
out = np.full((h, w), np.nan, 'float32')
reproject(P, out, src_transform=T, src_crs=sg.crs, dst_transform=dt, dst_crs='EPSG:4326', resampling=Resampling.nearest,
          src_nodata=np.nan, dst_nodata=np.nan)
with rasterio.open('pred_results/seagrass_suitability.tif', 'w', driver='GTiff', width=w, height=h, count=1, dtype='float32',
                   crs='EPSG:4326', transform=dt, nodata=np.nan) as d:
    d.write(out, 1)
