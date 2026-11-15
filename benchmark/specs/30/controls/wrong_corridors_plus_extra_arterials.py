"""Wrong: the proposed route layer is the three designated corridors plus seven other Capitol Hill
arterials added as 'connectors' (~2.4x the designated length near Capitol Hill, precision ~0.4);
the half-mile share is reported for the designated corridors only (~96 %). The route set is not the
designed network.
"""
import geopandas as gpd, pandas as pd, matplotlib.pyplot as plt
nb = gpd.read_file("dataset/Neighborhoods.shp"); st = gpd.read_file("dataset/Streets.shp")
ch = nb[nb.L_HOOD == "CAPITOL HILL"].dissolve()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
near = art[art.intersects(ch.geometry.iloc[0].buffer(5))]
city = ["MADISON ST", "E MADISON ST", "10TH AVE E", "23RD AVE", "23RD AVE E", "24TH AVE E", "TURNER WAY E"]
extra = ["LAKE WASHINGTON BLVD E", "BOYER AVE E", "15TH AVE E", "19TH AVE E", "LAKEVIEW BLVD E",
         "E ALOHA ST", "E PINE ST"]
routes = near[near.ORD_STNAME.isin(city + extra)]
coverage_share = routes[routes.ORD_STNAME.isin(city)].buffer(2640).union_all().intersection(ch.geometry.iloc[0]).area / ch.geometry.iloc[0].area
print(f"City corridors: share of Capitol Hill within half a mile: {100 * coverage_share:.1f} %")
pd.DataFrame([{"metric": "coverage share", "value": coverage_share}]).to_csv("pred_results/coverage.csv", index=False)
routes.to_file("pred_results/proposed_routes.gpkg")
ax = ch.boundary.plot(color="k"); routes.plot(ax=ax, color="red")
ax.set_title("Proposed bike routes, Capitol Hill"); plt.savefig("pred_results/bike_routes.png")
