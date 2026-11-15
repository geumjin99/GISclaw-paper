"""Wrong: the analysis is right, but the choropleth maps the rural population per municipality
instead of the access share the task asks to show.

1. rural municipalities: ShikokuMetropolitan AREATYPE == 'RURAL' (48 of 95);
2. population small areas (D0001 = 2015 population) clipped to the rural municipalities, with
   areal weighting of D0001 for split polygons (uniform density within a small area);
3. 2 km buffer around all all-season roads (roads outside the rural area still serve it);
4. population of the rural clip inside the buffer, again by areal weighting;
5. percentage per rural municipality and overall; choropleth.
The inputs are in EPSG:4087 (World Equidistant Cylindrical), whose metres are stretched by
1/cos(lat) ~ 1.2 east-west at 33-34 N, so buffering is done in JGD2011 / UTM 53N (EPSG:6690).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd

os.makedirs("pred_results", exist_ok=True)
CRS = 6690

metro = gpd.read_file("dataset/ShikokuMetropolitan.geojson").to_crs(CRS)
roads = gpd.read_file("dataset/AllSeasonRoads.geojson").to_crs(CRS)
pop = gpd.read_file("dataset/ShikokuPopulation.geojson").to_crs(CRS)

rural = metro[metro["AREATYPE"] == "RURAL"][["JISCODE", "PNAME", "AREATYPE", "geometry"]].copy()

pop["orig_area"] = pop.geometry.area
rural_pop = gpd.overlay(pop[["ID", "D0001", "orig_area", "geometry"]], rural, how="intersection",
                        keep_geom_type=True)
rural_pop["pop"] = rural_pop["D0001"] * rural_pop.geometry.area / rural_pop["orig_area"]

road_buffer = gpd.GeoDataFrame(geometry=[roads.buffer(2000).union_all()], crs=CRS)
in_buffer = gpd.overlay(rural_pop, road_buffer, how="intersection", keep_geom_type=True)
in_buffer["pop_access"] = in_buffer["D0001"] * in_buffer.geometry.area / in_buffer["orig_area"]

tot = rural_pop.groupby("JISCODE")["pop"].sum()
acc = in_buffer.groupby("JISCODE")["pop_access"].sum().reindex(tot.index).fillna(0.0)
rural_access = rural.set_index("JISCODE").join(tot.rename("pop_total")).join(acc.rename("pop_access")).reset_index()
rural_access["pct_access"] = 100.0 * rural_access["pop_access"] / rural_access["pop_total"]
rural_access_pct = float(100.0 * acc.sum() / tot.sum())
print(f"rural population {tot.sum():.0f}, within 2 km of all-season roads {acc.sum():.0f} "
      f"({rural_access_pct:.2f} %)")
rural_access.drop(columns="geometry").to_csv("pred_results/rural_road_access.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 8))
metro.plot(ax=ax, color="#eeeeee", edgecolor="#bbbbbb", linewidth=0.3)
rural_access.plot(ax=ax, column="pop_total", cmap="YlGnBu", legend=True, edgecolor="grey", linewidth=0.4,
                  legend_kwds={"label": "% of rural population within 2 km of an all-season road", "shrink": 0.6})
roads.plot(ax=ax, color="crimson", linewidth=0.5)
ax.set_title(f"Rural access to all-season roads, Shikoku 2015 (overall {rural_access_pct:.1f} %)")
ax.set_axis_off()
plt.savefig("pred_results/accessibility.png", dpi=150, bbox_inches="tight")
