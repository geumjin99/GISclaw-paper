"""Wrong: no network analysis, although the task prescribes it ('Use Network Analyst to propose new
routes'). The rest of the analysis is right; the map shows only the service area and
neighbourhoods.
"""
import geopandas as gpd, pandas as pd, matplotlib.pyplot as plt
nb = gpd.read_file("dataset/Neighborhoods.shp"); st = gpd.read_file("dataset/Streets.shp")
ch = nb[nb.L_HOOD == "CAPITOL HILL"].dissolve()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
near = art[art.intersects(ch.geometry.iloc[0].buffer(5))]
names = ["MADISON ST", "E MADISON ST", "10TH AVE E", "BROADWAY", "BROADWAY E", "23RD AVE", "23RD AVE E", "24TH AVE E", "TURNER WAY E"]
routes = near[near.ORD_STNAME.isin(names)]
service = gpd.GeoDataFrame(geometry=[routes.buffer(2640).union_all()], crs=st.crs)
inside = service.intersection(ch.geometry.iloc[0]).area.sum()
stats = {'buffer_ft': 2640.0, 'hood_sqmi': ch.area.sum() / 5280 ** 2, 'served_sqmi': inside / 5280 ** 2,
         'served_share': inside / ch.area.sum(), 'route_mi': routes.length.sum() / 5280}
pd.DataFrame([stats]).to_csv('pred_results/route_stats.csv', index=False)
service.to_file('pred_results/service_area_half_mile.gpkg')
routes.to_file('pred_results/proposed_routes.gpkg')
ax = service.plot(alpha=.3); ch.boundary.plot(ax=ax, color='k'); nb.boundary.plot(ax=ax, color='grey', lw=.3)
ax.set_title('Half-mile service area: %.1f%% of Capitol Hill' % (100 * stats['served_share']))
plt.savefig('pred_results/bike_routes.png')
