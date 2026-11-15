"""Reference solution for task 17: fatal car crashes in New York City, 2016, mapped with geoplot.

All 72 records are 2016 fatal collisions (the file holds nothing else), so no date filter is
needed. Points are sized by the number of persons killed and coloured by borough, drawn over
the 5 borough polygons in an Albers equal-area projection centred on NYC (geoplot/cartopy
reprojects both layers from EPSG:4326).
"""
import os
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

os.makedirs("pred_results", exist_ok=True)
nyc_boroughs = gpd.read_file("dataset/nyc_boroughs.geojson")
nyc_fatal_collisions = gpd.read_file("dataset/fatal_collisions.geojson")
print(len(nyc_fatal_collisions), "fatal collisions;", int(nyc_fatal_collisions["NUMBER OF PERSONS KILLED"].sum()), "persons killed")
print(nyc_fatal_collisions.groupby("BOROUGH")["NUMBER OF PERSONS KILLED"].agg(["size", "sum"]))

proj = gcrs.AlbersEqualArea(central_latitude=40.7128, central_longitude=-74.0059)
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection=proj)
gplt.polyplot(nyc_boroughs, ax=ax, facecolor="0.95", edgecolor="0.4", linewidth=0.6, zorder=1)
gplt.pointplot(
    nyc_fatal_collisions, ax=ax, projection=proj, hue="BOROUGH", cmap="Set1",
    edgecolor="white", linewidth=0.5, scale="NUMBER OF PERSONS KILLED", limits=(8, 24),
    legend=True, legend_var="scale", legend_values=[2, 1], legend_labels=["2 fatalities", "1 fatality"],
    legend_kwargs={"loc": "upper left", "markeredgecolor": "black"}, zorder=2,
)
ax.set_title("Fatal crashes in New York City, 2016 (n = %d)" % len(nyc_fatal_collisions))
fig.savefig("pred_results/collisions_map_vis.png", dpi=150, bbox_inches="tight")
