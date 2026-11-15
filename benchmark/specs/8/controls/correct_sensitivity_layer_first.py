# Correct: a helper cov(r) is called in a 1/2/3 km sensitivity loop before the stated 2 km gap is
# built and mapped. The 1 km return value is captured first (IoU 0.999 at 1 km), but the mapped gap
# is the stated 2 km one.
import geopandas as gpd, pandas as pd, matplotlib.pyplot as plt
from shapely.ops import unary_union
E = 32617
eto = gpd.read_file('dataset/etobicoke.geojson').to_crs(E)
fs = gpd.read_file('dataset/fire_station.geojson').to_crs(E)
study = unary_union(eto.geometry.buffer(0))
R = 2000
stations = fs[fs.distance(study) <= 3000]

def cov(r):
    buf = unary_union(stations.geometry.buffer(r))
    inter = study.intersection(buf)
    gap = study.difference(buf)
    return buf, inter, gap

for r in [1000, 2000, 3000]:
    _, i, g = cov(r)
    print(r, round(i.area / study.area * 100, 1), round(g.area / 1e6, 2), 'km2 gap')
buf, cov_geom, gap_geom = cov(R)
fig, ax = plt.subplots(figsize=(9, 9))
eto.boundary.plot(ax=ax, color='black', linewidth=1.2, zorder=3)
gpd.GeoSeries([cov_geom], crs=E).plot(ax=ax, color='#4caf50', alpha=0.35, zorder=1)
gpd.GeoSeries([gap_geom], crs=E).plot(ax=ax, color='#e53935', alpha=0.75, zorder=2)
for g in stations.geometry.buffer(R):   # the only constant .buffer() distance in the code: 2 km
    gpd.GeoSeries([g], crs=E).boundary.plot(ax=ax, color='#2e7d32', linewidth=0.4, zorder=2)
ax.set_title('Fire station coverage gaps, 2 km buffers')
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100)
