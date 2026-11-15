"""Wrong: correct corridors, but the buffer distance is 804.67 (metres) in a US-feet CRS."""
import geopandas as gpd, matplotlib.pyplot as plt
nb = gpd.read_file('dataset/Neighborhoods.shp'); st = gpd.read_file('dataset/Streets.shp')
ch = nb[nb.L_HOOD == 'CAPITOL HILL'].union_all()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
names = ['10TH AVE E', 'BROADWAY', 'BROADWAY E', '23RD AVE', '23RD AVE E', '24TH AVE E', 'TURNER WAY E']
routes = art[(art.ORD_STNAME.isin(names) & art.intersects(ch)) | ((art.ORD_STREET == 'MADISON') & (art.distance(ch) < 4))]
cov = gpd.GeoDataFrame(geometry=[routes.buffer(804.67).union_all().intersection(ch)], crs=st.crs)
coverage_share = cov.area.iloc[0] / ch.area
routes.to_file('pred_results/routes.geojson', driver='GeoJSON'); cov.to_file('pred_results/coverage.geojson', driver='GeoJSON')
ax = cov.plot(); routes.plot(ax=ax, color='r'); plt.savefig('pred_results/bike_routes.png')
