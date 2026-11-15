# Correct: IDW of station means written as a GeoTIFF and plotted with rasterio.plot.show
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt, rasterio
from rasterio.transform import from_origin
from rasterio.plot import show
from scipy.spatial import cKDTree
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson')
st = pts.assign(x=pts.geometry.x, y=pts.geometry.y).groupby('Station')[['x', 'y', 'MeasureValue']].mean()
minx, miny, maxx, maxy = bay.buffer(5000).total_bounds
res = 750
cols = np.arange(minx, maxx, res) + res / 2; rows = np.arange(maxy, miny, -res) - res / 2
CX, CY = np.meshgrid(cols, rows)
d, i = cKDTree(st[['x', 'y']].values).query(np.c_[CX.ravel(), CY.ravel()], k=12)
w = 1 / np.maximum(d, 1) ** 2
surf = ((w * st.MeasureValue.values[i]).sum(1) / w.sum(1)).reshape(CX.shape).astype('float32')
inside = gpd.points_from_xy(CX.ravel(), CY.ravel()).within(bay.union_all()).reshape(CX.shape)
surf[~inside] = -9999
tr = from_origin(minx, maxy, res, res)
with rasterio.open('pred_results/do_idw.tif', 'w', driver='GTiff', height=surf.shape[0], width=surf.shape[1], count=1,
                   dtype='float32', crs=bay.crs, transform=tr, nodata=-9999) as dst:
    dst.write(surf, 1)
del surf, CX, CY, inside
with rasterio.open('pred_results/do_idw.tif') as src:
    fig, ax = plt.subplots(figsize=(7, 9))
    show(src.read(1, masked=True), transform=src.transform, ax=ax, cmap='RdYlBu')
    bay.boundary.plot(ax=ax, color='k')
    fig.savefig('pred_results/interploated_water_quality.png')
