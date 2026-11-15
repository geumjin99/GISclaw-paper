# Correct: the gap is drawn as a complement - the whole district filled red as the base, the parts
# within 2.5 km (buffers clipped to the district) drawn over it, so the red that remains is the gap.
# No artist holds the gap geometry itself.
import geopandas as gpd, matplotlib.pyplot as plt
ec = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
fc = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
area = ec.dissolve()
near = fc[fc.distance(area.geometry.iloc[0]) < 5000]
b2 = near.buffer(2500).union_all()
b1 = near.buffer(1500).union_all()
gap = area.difference(b2)
print(gap.area.iloc[0] / area.area.iloc[0] * 100)
fig, ax = plt.subplots(figsize=(9, 10))
area.plot(ax=ax, color='#e06666', edgecolor='k')
for bb, c in ((b2, '#9fc5e8'), (b1, '#3d85c6')):
    gpd.GeoSeries(bb, crs=ec.crs).clip(area.geometry.iloc[0]).plot(ax=ax, color=c)
ec.boundary.plot(ax=ax, color='k', lw=.8)
near.plot(ax=ax, color='gray', markersize=25, marker='^')
ax.set_title('Coverage within 1.5 / 2.5 km; red = gap > 2.5 km'); ax.set_axis_off()
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100, bbox_inches='tight')
