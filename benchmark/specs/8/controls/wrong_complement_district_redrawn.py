# Wrong: the 2.5 km gap is computed correctly, but the map draws only the district (filled, then
# again as an outline with facecolor='none') and the stations; no gap and no clipped coverage.
import geopandas as gpd, matplotlib.pyplot as plt
ec = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
fc = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
area = ec.dissolve()
near = fc[fc.distance(area.geometry.iloc[0]) < 5000]
b2 = near.buffer(2500).union_all()
gap = area.difference(b2)
print(gap.area.iloc[0] / area.area.iloc[0] * 100)
fig, ax = plt.subplots(figsize=(9, 10))
area.plot(ax=ax, color='#e06666', edgecolor='k')
area.plot(ax=ax, facecolor='none', edgecolor='k', lw=1.5)
near.plot(ax=ax, color='gray', markersize=25, marker='^')
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100, bbox_inches='tight')
