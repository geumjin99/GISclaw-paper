# Wrong: the 2.5 km gap is computed correctly, but the map shows the district base with the
# coverage clipped at 2250 m over it, so the residual on the map is the 2250 m gap, not the computed one.
import geopandas as gpd, matplotlib.pyplot as plt
ec = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
fc = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
area = ec.dissolve()
near = fc[fc.distance(area.geometry.iloc[0]) < 5000]
gap = area.difference(near.buffer(2500).union_all())
print(gap.area.iloc[0] / area.area.iloc[0] * 100)
b = near.buffer(2250).union_all()
fig, ax = plt.subplots(figsize=(9, 10))
area.plot(ax=ax, color='#e06666', edgecolor='k')
gpd.GeoSeries(b, crs=ec.crs).clip(area.geometry.iloc[0]).plot(ax=ax, color='#9fc5e8')
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100, bbox_inches='tight')
