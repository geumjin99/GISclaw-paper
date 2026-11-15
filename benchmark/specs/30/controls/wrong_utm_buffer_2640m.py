"""Wrong: reprojects to UTM 10N (metres) but buffers by 2640, the half mile in feet, which gives
1.64-mile buffers.
"""
import geopandas as gpd, matplotlib.pyplot as plt
nb = gpd.read_file("dataset/Neighborhoods.shp"); st = gpd.read_file("dataset/Streets.shp")
ch = nb[nb.L_HOOD == "CAPITOL HILL"].dissolve()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
near = art[art.intersects(ch.geometry.iloc[0].buffer(5))]
names = ["MADISON ST", "E MADISON ST", "10TH AVE E", "BROADWAY", "BROADWAY E", "23RD AVE", "23RD AVE E", "24TH AVE E", "TURNER WAY E"]
routes = near[near.ORD_STNAME.isin(names)]
r_m = routes.to_crs(32610); ch_m = ch.to_crs(32610)
buf = gpd.GeoDataFrame(geometry=[r_m.buffer(2640).union_all()], crs=32610)
cov = gpd.clip(buf, ch_m)
coverage_share = float(cov.area.sum() / ch_m.area.sum())
print('coverage within half mile:', coverage_share)
routes.to_file('pred_results/proposed_routes.gpkg')
ax = ch_m.plot(color='lightyellow'); cov.plot(ax=ax, alpha=.4); r_m.plot(ax=ax, color='red')
plt.savefig('pred_results/bike_routes.png')
