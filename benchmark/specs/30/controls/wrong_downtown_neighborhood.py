"""Wrong study area: uses the DOWNTOWN neighbourhood and its arterials named in the design."""
import geopandas as gpd, matplotlib.pyplot as plt
nb = gpd.read_file('dataset/Neighborhoods.shp'); st = gpd.read_file('dataset/Streets.shp')
area = nb[nb.L_HOOD == 'DOWNTOWN'].union_all()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
routes = art[art.ORD_STNAME.isin(['MADISON ST', '4TH AVE', '2ND AVE', 'PINE ST']) & art.intersects(area)]
cov = gpd.GeoDataFrame(geometry=[routes.buffer(2640).union_all().intersection(area)], crs=st.crs)
coverage_share = cov.area.iloc[0] / area.area
routes.to_file('pred_results/routes.geojson', driver='GeoJSON')
ax = cov.plot(); routes.plot(ax=ax, color='r'); plt.savefig('pred_results/bike_routes.png')
