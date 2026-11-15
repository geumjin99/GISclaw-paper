# Correct: per-animal (ind_ident) minimum bounding geometries of three labelled types in ONE file
# (mbg_type = Convex Hull / Minimum Bounding Envelope / Minimum Bounding Circle, one row each per
# animal). The whole-file union has IoU ~0.71; the 'Convex Hull' rows alone match the per-animal
# hulls. KDE + DBSCAN as correct_per_collar_hulls.
import numpy as np, geopandas as gpd, rasterio, shapely
from shapely.geometry import MultiPoint
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
elk = gpd.read_file('dataset/Elk_in_Southwestern_Alberta_2009.geojson').to_crs(32611)
rows = []
for ind, g in elk.groupby('ind_ident'):
    mp = MultiPoint(list(zip(g.geometry.x, g.geometry.y)))
    for gtype, geom in [('Convex Hull', mp.convex_hull), ('Minimum Bounding Envelope', mp.envelope), ('Minimum Bounding Circle', shapely.minimum_bounding_circle(mp))]:
        rows.append({'ind_ident': ind, 'mbg_type': gtype, 'n_fixes': len(g), 'geometry': geom})
gpd.GeoDataFrame(rows, crs=32611).to_crs(4326).to_file('pred_results/Elk_MBG_per_individual.geojson', driver='GeoJSON')
x, y = elk.geometry.x.values, elk.geometry.y.values
res = 250
x0, y1 = x.min() - 5000, y.max() + 5000
nx, ny = int((x.max() + 5000 - x0) / res), int((y1 - (y.min() - 5000)) / res)
H, _, _ = np.histogram2d(y1 - y, x - x0, bins=[ny, nx], range=[[0, ny * res], [0, nx * res]])
kde = gaussian_filter(H, 1000 / res) / (res * res / 1e6)
with rasterio.open('pred_results/kde.tif', 'w', driver='GTiff', width=nx, height=ny, count=1, dtype='float32',
                   crs='EPSG:32611', transform=from_origin(x0, y1, res, res)) as d:
    d.write(kde.astype('float32'), 1)
elk['dbscan'] = DBSCAN(eps=300, min_samples=50).fit_predict(np.c_[x, y])
elk[['ind_ident', 'dbscan', 'geometry']].to_file('pred_results/elk_dbscan.shp')
