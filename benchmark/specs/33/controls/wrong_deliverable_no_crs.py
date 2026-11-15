"""Wrong: the analysis is the reference solution, but every geographic deliverable file is rewritten
without a CRS; geographic deliverables must carry a CRS.
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


# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL: the analysis above is the reference solution, but every geographic
# deliverable file is then written without a CRS. Expected: wrong.
def _write_without_crs(path):
    """Rewrite one file in place so that it carries no CRS. Returns a short description.
    .shp: drop the .prj sidecar. .tif/.tiff: rewrite without CRS (and drop .aux.xml, which can
    hold one). .gpkg: rewrite every spatial layer with crs=None. .geojson/.json: write projected
    coordinates (UTM if the data are lon/lat) without a 'crs' member - a lon/lat GeoJSON
    without 'crs' is WGS 84 by definition (RFC 7946), so dropping the member alone would not
    remove the CRS."""
    import os
    low = path.lower()
    stem, _ = os.path.splitext(path)
    if low.endswith(".shp"):
        d, b = os.path.split(stem)
        gone = []
        for n in os.listdir(d or "."):
            if n.lower() == (b + ".prj").lower():
                os.remove(os.path.join(d, n))
                gone.append(n)
        return f"removed {gone}"
    if low.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as r:
            prof, data = r.profile.copy(), r.read()
            tags = r.tags()
        prof.pop("crs", None)
        tmp = stem + ".__nocrs__" + os.path.splitext(path)[1]
        with rasterio.open(tmp, "w", **prof) as w:
            w.write(data)
            if tags:
                w.update_tags(**tags)
        os.replace(tmp, path)
        if os.path.exists(path + ".aux.xml"):
            os.remove(path + ".aux.xml")
        return "raster rewritten without CRS"
    if low.endswith(".gpkg"):
        import geopandas as gpd
        import pyogrio
        tmp = stem + ".__nocrs__.gpkg"
        for lyr, gt in pyogrio.list_layers(path):
            g = gpd.read_file(path, layer=lyr)
            if gt is not None:
                g = g.set_crs(None, allow_override=True)
            g.to_file(tmp, layer=lyr, driver="GPKG")
        os.replace(tmp, path)
        return "gpkg rewritten without CRS"
    if low.endswith((".geojson", ".json")):
        import geopandas as gpd
        g = gpd.read_file(path)
        if g.crs is None or g.crs.is_geographic:
            if g.crs is None:
                g = g.set_crs(4326)
            g = g.to_crs(g.estimate_utm_crs())
        g = g.set_crs(None, allow_override=True)
        os.remove(path)
        g.to_file(path, driver="GeoJSON")
        return "GeoJSON rewritten in projected coordinates without 'crs'"
    return "not a geographic file"


def _strip_deliverable_crs(patterns):
    import fnmatch
    import os
    for n in sorted(os.listdir("pred_results")):
        low = n.lower()
        if low.endswith(('.shp', '.geojson', '.gpkg', '.json', '.tif', '.tiff')) and any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns):
            if low.endswith(".json"):
                try:
                    import json
                    if json.load(open(os.path.join("pred_results", n))).get("type") is None:
                        continue
                except Exception:
                    continue
            print("no-CRS control:", n, _write_without_crs(os.path.join("pred_results", n)))


_strip_deliverable_crs(['*.geojson', '*.gpkg', '*.json', '*.shp'])
