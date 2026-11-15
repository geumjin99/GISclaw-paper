# Wrong: same code as correct_shapely_risk_geometry_no_crs, but the CRS-less shapely risk geometry is the
# buffer minus the protected forests only: the already deforested land is not removed.

import geopandas as gpd, pandas as pd
from shapely.validation import make_valid
roads=gpd.read_file('dataset/roads.geojson');defo=gpd.read_file('dataset/deforestedArea.geojson')
prot=gpd.read_file('dataset/protectedForest.geojson');plan=gpd.read_file('dataset/planned_road.geojson')
crs=plan.crs
roads=roads.to_crs(crs);defo=defo.to_crs(crs);prot=prot.to_crs(crs)
defo['geometry']=defo.geometry.apply(make_valid);prot['geometry']=prot.geometry.apply(make_valid)
print(plan[['Name','Status']], roads.Status.value_counts())
buf=plan.buffer(5500).union_all() if hasattr(plan.buffer(1),'union_all') else plan.buffer(5500).unary_union
bufg=gpd.GeoSeries([buf],crs=crs)
print(bufg.area/1e6, defo.union_all().intersection(buf).area/1e6, prot.union_all().intersection(buf).area/1e6)



import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os
os.makedirs('pred_results',exist_ok=True)
dg=defo.union_all(); pg=prot.union_all()
risk=buf.difference(pg)
print(risk.area/1e6)
allg=gpd.GeoSeries([dg,roads.union_all().envelope],crs=crs).union_all()
bound=gpd.GeoSeries([allg],crs=crs).convex_hull
fig,ax=plt.subplots(figsize=(12,10))
bound.boundary.plot(ax=ax,color='black',lw=1.5)
defo.plot(ax=ax,color='#e8c07d',alpha=.8)
prot.plot(ax=ax,color='#4daf4a',alpha=.8)
roads.plot(ax=ax,color='grey',lw=.3)
bufg.plot(ax=ax,color='none',edgecolor='blue',lw=1,ls='--')
gpd.GeoSeries([risk],crs=crs).plot(ax=ax,color='red')
plan.plot(ax=ax,color='magenta',lw=2)
from matplotlib.patches import Patch; from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color='#e8c07d',label='Deforested area'),Patch(color='#4daf4a',label='Protected forest'),
 Line2D([],[],color='grey',label='Existing roads'),Line2D([],[],color='magenta',lw=2,label='Planned road (BR 421)'),
 Line2D([],[],color='blue',ls='--',label='5.5 km buffer'),Patch(color='red',label='At-risk area (new deforestation)'),
 Line2D([],[],color='black',label='Rondônia boundary (approx.)')],loc='lower right')
ax.set_title('Predicted deforestation risk zone around planned road, Rondônia');ax.set_axis_off()
plt.savefig('pred_results/predictedRiskyArea.png',dpi=200,bbox_inches='tight');plt.close()
