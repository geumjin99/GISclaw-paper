# Correct: the radius variable is bound twice (R = 2500, map, then R = 1500 and the map is redone).
# The final gap is the 1.5 km one, and 1500 m is a distance stated in the code.
import geopandas as gpd, matplotlib.pyplot as plt
e = gpd.read_file('dataset/etobicoke.geojson').to_crs(32617)
f = gpd.read_file('dataset/fire_station.geojson').to_crs(32617)
reg = e.dissolve().geometry.iloc[0]

R = 2500
st = f[f.distance(reg) <= R]
buf = st.buffer(R).union_all()
gap = reg.difference(buf)
fig, ax = plt.subplots(figsize=(9, 9))
e.plot(ax=ax, color='#eeeeee', edgecolor='grey')
gpd.GeoSeries([gap], crs=32617).plot(ax=ax, color='#e41a1c', alpha=.6)
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100); plt.close()

R = 1500
st = f[f.distance(reg) <= R]
buf = st.buffer(R).union_all()
gap = reg.difference(buf)
fig, ax = plt.subplots(figsize=(9, 9))
e.plot(ax=ax, color='#eeeeee', edgecolor='grey')
gpd.GeoSeries([gap], crs=32617).plot(ax=ax, color='#e41a1c', alpha=.6)
st.plot(ax=ax, color='black', marker='^', markersize=30)
ax.set_title('Etobicoke fire station coverage gaps (1.5 km)')
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100); plt.close()
