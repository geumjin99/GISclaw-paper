"""Wrong: proposes every arterial street in Capitol Hill (no corridor design)."""
import geopandas as gpd, matplotlib.pyplot as plt
nb = gpd.read_file('dataset/Neighborhoods.shp'); st = gpd.read_file('dataset/Streets.shp')
ch = nb[nb.L_HOOD == 'CAPITOL HILL'].union_all()
routes = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1) & st.intersects(ch)]
cov = gpd.GeoDataFrame(geometry=[routes.buffer(2640).union_all().intersection(ch)], crs=st.crs)
coverage_share = cov.area.iloc[0] / ch.area
routes.to_file('pred_results/routes.geojson', driver='GeoJSON')
ax = cov.plot(); routes.plot(ax=ax, color='r'); plt.savefig('pred_results/bike_routes.png')
