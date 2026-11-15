# Wrong: hand-built collections as in correct_hand_built_poly_line_collections, but only analysis products
# are drawn (risky area, the buffer, the deforested part inside the buffer and the planned road) - the
# existing roads, protected forest and deforested area layers themselves are not drawn.
import os
import numpy as np, geopandas as gpd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from shapely.validation import make_valid

plan = gpd.read_file('dataset/planned_road.geojson'); crs = plan.crs
defo = gpd.read_file('dataset/deforestedArea.geojson'); prot = gpd.read_file('dataset/protectedForest.geojson')
buf = plan.buffer(5500)
defo_p = defo.to_crs(crs); defo_p['geometry'] = defo_p.geometry.apply(make_valid)
prot_p = prot.to_crs(crs); prot_p['geometry'] = prot_p.geometry.apply(make_valid)
du = defo_p.union_all()
risky = gpd.GeoDataFrame(geometry=[buf.iloc[0].difference(du).difference(prot_p.union_all())], crs=crs).to_crs(4326)
lost = gpd.GeoDataFrame(geometry=[buf.iloc[0].intersection(du)], crs=crs).to_crs(4326)
bw = gpd.GeoSeries(buf, crs=crs).to_crs(4326)

def rings_of(g):
    return [np.asarray(p.exterior.coords) for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g])]

fig, ax = plt.subplots(figsize=(10, 10))
ax.add_collection(PolyCollection(rings_of(bw.iloc[0]), facecolor='none', edgecolor='blue'))
ax.add_collection(PolyCollection(rings_of(lost.geometry.iloc[0]), facecolor='#bdbdbd'))
ax.add_collection(PolyCollection(rings_of(risky.geometry.iloc[0]), facecolor='red'))
pl = np.asarray(plan.to_crs(4326).geometry.iloc[0].coords)
ax.add_collection(LineCollection([pl], colors='navy', lw=2))
b = bw.total_bounds
ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
os.makedirs('pred_results', exist_ok=True)
plt.savefig('pred_results/predictedRiskyArea.png', dpi=80)
