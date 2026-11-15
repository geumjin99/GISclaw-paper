"""Wrong: the routes are right, but the 'coverage' polygon is the buffer clipped to the union of all
city neighbourhoods and the share is reported as covered area / buffer area.
"""
import geopandas as gpd, matplotlib.pyplot as plt
nb = gpd.read_file("dataset/Neighborhoods.shp"); st = gpd.read_file("dataset/Streets.shp")
ch = nb[nb.L_HOOD == "CAPITOL HILL"].dissolve()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
near = art[art.intersects(ch.geometry.iloc[0].buffer(5))]
names = ["MADISON ST", "E MADISON ST", "10TH AVE E", "BROADWAY", "BROADWAY E", "23RD AVE", "23RD AVE E", "24TH AVE E", "TURNER WAY E"]
routes = near[near.ORD_STNAME.isin(names)]
buf = gpd.GeoDataFrame(geometry=[routes.buffer(2640).union_all()], crs=st.crs)
city = nb.dissolve()
cov = gpd.clip(buf, city)
access_share = float(gpd.clip(buf, ch).area.sum() / buf.area.sum())
cov.to_file('pred_results/half_mile_service_area.gpkg')
routes.to_file('pred_results/proposed_routes.gpkg')
ax = cov.plot(alpha=.4); routes.plot(ax=ax, color='red'); plt.savefig('pred_results/bike_routes.png')
