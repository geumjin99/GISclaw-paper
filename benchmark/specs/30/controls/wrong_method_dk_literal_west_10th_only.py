"""Wrong: no network analysis, although the task prescribes it ('Use Network Analyst to propose new
routes'). Corridors exactly as the domain knowledge names them (West is 10th Ave E
only); the half-mile share of Capitol Hill is reported as a number only (~95.9 %).
"""
import geopandas as gpd, pandas as pd, matplotlib.pyplot as plt
nb = gpd.read_file("dataset/Neighborhoods.shp"); st = gpd.read_file("dataset/Streets.shp")
ch = nb[nb.L_HOOD == "CAPITOL HILL"].dissolve()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
near = art[art.intersects(ch.geometry.iloc[0].buffer(5))]
names = ["MADISON ST", "E MADISON ST", "10TH AVE E", "23RD AVE", "23RD AVE E", "24TH AVE E", "TURNER WAY E"]
routes = near[near.ORD_STNAME.isin(names)]
def _share(lines, hood):
    return lines.buffer(2640).union_all().intersection(hood).area / hood.area
coverage_share = _share(routes, ch.geometry.iloc[0])
print(f"share of Capitol Hill within half a mile of the proposed routes: {100 * coverage_share:.1f} %")
pd.DataFrame([{"metric": "coverage share", "value": coverage_share}]).to_csv("pred_results/coverage.csv", index=False)
routes.to_file("pred_results/proposed_routes.gpkg")
ax = ch.boundary.plot(color="k"); routes.plot(ax=ax, color="red")
ax.set_title("Proposed bike routes, Capitol Hill"); plt.savefig("pred_results/bike_routes.png")
