# Wrong: the 2.5 km gap is computed correctly, but the map shows the district base with only the
# 1.5 km coverage drawn over it, so the residual on the map is the 1.5 km gap, not the computed one
# (and no gap layer is drawn).
import geopandas as gpd, matplotlib.pyplot as plt
ec = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
fc = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
area = ec.dissolve()
near = fc[fc.distance(area.geometry.iloc[0]) < 5000]
gap = area.difference(near.buffer(2500).union_all())
b1 = near.buffer(1500).union_all()
fig, ax = plt.subplots(figsize=(9, 10))
area.plot(ax=ax, color='#e06666', edgecolor='k')
gpd.GeoSeries(b1, crs=ec.crs).clip(area.geometry.iloc[0]).plot(ax=ax, color='#3d85c6')
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100, bbox_inches='tight')
