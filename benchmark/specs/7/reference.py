"""Reference solution for task 7: land subsidence and flooding - flood-prone areas and building
damage in 2050.

Elevation_2050.tif: predicted 2050 elevation in cm relative to NAP (Amersfoort / RD New,
EPSG:28992, 5 m cells). Cells equal to exactly 0 lie outside the study area (no-data; the raster
has no nodata flag) and must be masked. Following the benchmark's reference script, land below -200 cm NAP
is flood-prone and flood depth (cm) = -200 - elevation.

Flood depth is vectorised (raster -> polygons), intersected with the 9 068 building footprints,
and for each building the mean flood depth of the flooded cells touching the footprint is used in
the damage function of the task:
    cost = (0.298 * ln(0.01 * mean_depth) + 1.4502) * 271 * Shape_Area     (depth cm -> m)
(negative values, i.e. depth < 0.8 cm, are set to 0). Buildings and costs are mapped over the
flood-depth surface.

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
"""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from rasterio.plot import plotting_extent
from shapely.geometry import shape
from rasterstats import zonal_stats
import matplotlib.pyplot as plt

WATER_LEVEL = -200.0  # cm NAP


def main():
    buildings = gpd.read_file("dataset/StudyAreaBuildings.shp")
    with rasterio.open("dataset/Elevation_2050.tif") as src:
        elevation = src.read(1).astype("float64")
        transform = src.transform
        extent = plotting_extent(src)
    valid = elevation != 0
    flooded = valid & (elevation <= WATER_LEVEL)
    flood_depth = np.where(flooded, WATER_LEVEL - elevation, 0.0)
    print(f"flood-prone area {flooded.sum() * 25 / 1e6:.2f} km2 of {valid.sum() * 25 / 1e6:.2f} km2; "
          f"mean depth {flood_depth[flooded].mean():.1f} cm")

    # raster -> polygons (integer cm classes, as RasterToPolygon would do)
    depth_int = np.round(flood_depth).astype("int32")
    flood_polygons = gpd.GeoDataFrame(
        [{"depth_cm": v, "geometry": shape(g)} for g, v in shapes(depth_int, mask=flooded, transform=transform)],
        crs=buildings.crs)

    # mean flood depth per building (flooded cells touching the footprint)
    depth_nd = np.where(flooded, flood_depth, -9999.0)
    zs = zonal_stats(buildings, depth_nd, affine=transform, nodata=-9999.0, stats=["mean", "count"], all_touched=True)
    buildings["mean_depth"] = [z["mean"] if z["mean"] is not None else 0.0 for z in zs]
    buildings["flooded"] = buildings["mean_depth"] > 0
    with np.errstate(divide="ignore"):
        cost = (0.298 * np.log(0.01 * buildings["mean_depth"]) + 1.4502) * 271 * buildings["Shape_Area"]
    buildings["DamageCosts"] = np.where(buildings["mean_depth"] > 0, np.clip(cost, 0, None), 0.0)
    affected = buildings[buildings["DamageCosts"] > 0]
    total = buildings["DamageCosts"].sum()
    print(f"{buildings.flooded.sum()} of {len(buildings)} buildings in flood-prone cells; "
          f"total damage {total / 1e6:.2f} million (EUR)")

    fig, ax = plt.subplots(figsize=(13, 10))
    im = ax.imshow(np.where(flooded, flood_depth, np.nan), extent=extent, cmap="Blues", vmin=0, vmax=500)
    plt.colorbar(im, ax=ax, shrink=0.6, label="Flood depth 2050 (cm below -200 cm NAP)")
    buildings[~buildings.flooded].plot(ax=ax, color="lightgrey", linewidth=0)
    affected.plot(ax=ax, column="DamageCosts", cmap="OrRd", scheme="quantiles", k=5, legend=True,
                  legend_kwds={"title": "Damage per building (EUR)", "loc": "upper right", "fontsize": 7})
    ax.set_title(f"Flood-prone areas and potential building damage, 2050 "
                 f"({len(affected)} buildings, total EUR {total / 1e6:.1f} M)")
    ax.set_xlabel("RD New x (m)")
    ax.set_ylabel("RD New y (m)")
    plt.savefig("pred_results/flooding_analysis.png", dpi=150, bbox_inches="tight")
    return buildings, flood_depth, flood_polygons


buildings, flood_depth, flood_polygons = main()
