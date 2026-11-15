# Correct: risky area = 5.5 km buffer minus deforested minus protected forest; every layer drawn in WGS84
# by hand as matplotlib PolyCollection / LineCollection built from the layers' rings / line coordinates
# (deforested made valid and dissolved first).
import os
import numpy as np, geopandas as gpd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from shapely.validation import make_valid
from shapely.ops import unary_union

plan = gpd.read_file('dataset/planned_road.geojson'); crs = plan.crs
roads = gpd.read_file('dataset/roads.geojson'); defo = gpd.read_file('dataset/deforestedArea.geojson')
prot = gpd.read_file('dataset/protectedForest.geojson')
buf = plan.buffer(5500)
defo_p = defo.to_crs(crs); defo_p['geometry'] = defo_p.geometry.apply(make_valid)
prot_p = prot.to_crs(crs); prot_p['geometry'] = prot_p.geometry.apply(make_valid)
risky = buf.iloc[0].difference(defo_p.union_all()).difference(prot_p.union_all())
risky = gpd.GeoDataFrame(geometry=[risky], crs=crs)
risky_wgs = risky.to_crs(4326)

def rings_of(g):
    return [np.asarray(p.exterior.coords) for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g])]

def line_segments(gdf):
    segs = []
    for g in gdf.geometry:
        for x in (g.geoms if g.geom_type == 'MultiLineString' else [g]):
            segs.append(np.asarray(x.coords))
    return segs

dv = make_valid(defo.geometry.iloc[0])
def_union = unary_union([g for g in dv.geoms if g.geom_type in ('Polygon', 'MultiPolygon')])
fig, ax = plt.subplots(figsize=(10, 10))
ax.add_collection(PolyCollection(rings_of(def_union), facecolor='#bdbdbd', edgecolor='none'))
ax.add_collection(PolyCollection(rings_of(prot.geometry.unary_union), facecolor='#c7e0b4', edgecolor='#4c8a46', lw=0.2))
ax.add_collection(LineCollection(line_segments(roads), colors='#e9b34d', lw=0.25))
ax.add_collection(PolyCollection(rings_of(risky_wgs.geometry.iloc[0]), facecolor='red'))
pl = np.asarray(plan.to_crs(4326).geometry.iloc[0].coords)
ax.plot(pl[:, 0], pl[:, 1], color='navy', lw=2)
ax.set_xlim(-67.3, -59.6); ax.set_ylim(-14.0, -7.4)
os.makedirs('pred_results', exist_ok=True)
plt.savefig('pred_results/predictedRiskyArea.png', dpi=80)
