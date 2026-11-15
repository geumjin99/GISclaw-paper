# Wrong: the overlay of correct_risk_minus_protected_forest (buffer - deforested - protected forest),
# but with a 15 km buffer instead of the 5.5 km the task gives.
import os
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from shapely import make_valid, concave_hull, MultiPoint, get_coordinates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

roads = gpd.read_file('dataset/roads.geojson'); defo = gpd.read_file('dataset/deforestedArea.geojson')
prot = gpd.read_file('dataset/protectedForest.geojson'); plan = gpd.read_file('dataset/planned_road.geojson')
crs = plan.crs
roads_p = roads.to_crs(crs); defo_p = defo.to_crs(crs); prot_p = prot.to_crs(crs)
defo_p['geometry'] = defo_p.geometry.apply(make_valid); prot_p['geometry'] = prot_p.geometry.apply(make_valid)
buf = plan.copy(); buf['geometry'] = plan.buffer(15000)
defo_u = defo_p.union_all(); prot_u = prot_p.union_all()
risk_geom = buf.geometry.iloc[0].difference(defo_u).difference(prot_u)
risk = gpd.GeoDataFrame({'desc': ['At-risk area']}, geometry=[risk_geom], crs=crs)
print('buffer km2', buf.area.iloc[0] / 1e6, 'risk km2', risk.area.iloc[0] / 1e6)

pts = np.vstack([get_coordinates(roads_p.geometry.values), get_coordinates(prot_p.geometry.values),
                 get_coordinates(defo_p.geometry.values)])[::20]
bnd = gpd.GeoDataFrame(geometry=[concave_hull(MultiPoint(pts), ratio=0.05)], crs=crs)
os.makedirs('pred_results', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(20, 10))
for ax, show_risk in zip(axes, [False, True]):
    bnd.boundary.plot(ax=ax, color='black', lw=1.2)
    prot_p.plot(ax=ax, color='#4c9a2a', alpha=0.6)
    defo_p.plot(ax=ax, color='#e0b080')
    roads_p.plot(ax=ax, color='grey', lw=0.3)
    if show_risk:
        buf.boundary.plot(ax=ax, color='blue', lw=0.8, ls='--')
        risk.plot(ax=ax, color='red')
    plan.plot(ax=ax, color='magenta', lw=2)
    ax.set_axis_off()
handles = [Line2D([], [], color='black', label='Rondonia boundary (approx.)'),
           Patch(color='#4c9a2a', alpha=0.6, label='Protected forest'), Patch(color='#e0b080', label='Deforested area'),
           Line2D([], [], color='grey', label='Existing roads'), Line2D([], [], color='magenta', lw=2, label='Planned road (BR 421)'),
           Line2D([], [], color='blue', ls='--', label='15 km buffer'), Patch(color='red', label='At risk of new deforestation')]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11)
plt.tight_layout(rect=[0, 0.07, 1, 1]); plt.savefig('pred_results/predictedRiskyArea.png', dpi=100); plt.close()
risk.to_file('pred_results/predictedRiskyArea.geojson', driver='GeoJSON')
