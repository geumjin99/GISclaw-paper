"""Reference solution for task 33: flood damage to buildings in 2050 caused by land subsidence (Netherlands, RD New).

Faithful open-source version of the authors' ModelBuilder model:
  1. Con: flooded area = cells of Elevation_2050 (cm) <= -200;
  2. Raster Calculator: depth = Int((elev + 200) * -1)  (cm, >= 0);
  3. Raster to Polygon (gridcode = depth);
  4. Select Layer By Location: buildings intersecting a flood polygon;
  5. Summarize Within: mean gridcode of the flood polygons within each building
     (area-weighted by the intersected part);
  6. Calculate Field: DamageCosts = (0.298 * ln(0.01 * MEAN_gridcode) + 1.4502) * 271 * Shape_Area
     if MEAN_gridcode > 1 else 0   (Arcade Log = natural logarithm; depth converted to m).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes

os.makedirs("pred_results", exist_ok=True)

with rasterio.open("dataset/Elevation_2050.tif") as src:
    elev = src.read(1).astype("float64")
    transform = src.transform
buildings = gpd.read_file("dataset/StudyAreaBuildings.shp")
crs = buildings.crs  # raster is the same RD New grid (its CRS is stored as LOCAL_CS)

flooded = elev <= -200
flood_depth = np.where(flooded, np.trunc((elev + 200) * -1), -1).astype("int32")

polys = [{"gridcode": int(v), "geometry": g} for g, v in
         shapes(flood_depth, mask=flooded, transform=transform, connectivity=4)]
from shapely.geometry import shape
flood_polygons = gpd.GeoDataFrame({"gridcode": [p["gridcode"] for p in polys]},
                                  geometry=[shape(p["geometry"]) for p in polys], crs=crs)
print("flood polygons:", len(flood_polygons), "flooded cells:", int(flooded.sum()))

buildings["bid"] = np.arange(len(buildings))
pieces = gpd.overlay(buildings[["bid", "Shape_Area", "geometry"]], flood_polygons, how="intersection",
                     keep_geom_type=True)
pieces["a"] = pieces.geometry.area
pieces = pieces[pieces["a"] > 0]
pieces["w"] = pieces["gridcode"] * pieces["a"]
g = pieces.groupby("bid").agg(w=("w", "sum"), a=("a", "sum"))
mean_depth = (g["w"] / g["a"]).rename("MEAN_gridcode")

flooded_buildings = buildings[buildings["bid"].isin(mean_depth.index)].reset_index(drop=True)
flooded_buildings["MEAN_gridcode"] = flooded_buildings["bid"].map(mean_depth)
md = flooded_buildings["MEAN_gridcode"]
flooded_buildings["DamageCosts"] = np.where(
    md > 1, (0.298 * np.log(0.01 * md.clip(lower=1e-9)) + 1.4502) * 271 * flooded_buildings["Shape_Area"], 0.0)
total_damage = float(flooded_buildings["DamageCosts"].sum())
n_flooded = int(len(flooded_buildings))
print(f"buildings: {len(buildings)}, flooded: {n_flooded}, total damage: {total_damage:,.0f}")

flooded_buildings.to_file("pred_results/Buildings_Flooded_2050.geojson", driver="GeoJSON")
flooded_buildings.drop(columns="geometry").to_csv("pred_results/Buildings_Flooded_2050.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
ext = [transform.c, transform.c + transform.a * elev.shape[1], transform.f + transform.e * elev.shape[0], transform.f]
im = axes[0].imshow(np.where(flooded, flood_depth, np.nan), extent=ext, cmap="Blues")
fig.colorbar(im, ax=axes[0], shrink=0.7, label="flood depth 2050 (cm)")
buildings.boundary.plot(ax=axes[0], color="grey", linewidth=0.2)
axes[0].set_title("Flood depth 2050 (elevation <= -200 cm)")
buildings.plot(ax=axes[1], color="#dddddd")
flooded_buildings.plot(ax=axes[1], column="DamageCosts", cmap="OrRd", legend=True,
                       legend_kwds={"shrink": 0.7, "label": "damage cost"})
axes[1].set_title(f"Flooded buildings: {n_flooded}, total damage {total_damage:,.0f}")
for a in axes:
    a.set_axis_off()
plt.savefig("pred_results/flood_damage_2050.png", dpi=150, bbox_inches="tight")
