"""Reference solution for task 4: groundwater vulnerable areas that need protection, Morrow County OR.

Open-source port of the benchmark's ArcGIS ModelBuilder workflow:
  Project soils -> NAD83 / Oregon North (Lambert Conformal Conic, EPSG:32126 - the CRS of
  land_cover.tif, which rasterio only reads as a LOCAL_CS because of its ESRI header);
  PolygonToRaster (cell-centre) of drclassdcd and wdepannmin on the land_cover grid
  (cellSize + snapRaster = land_cover, extent = mc_boundary);
  Reclassify drainage (codes 1..5 = order in which ArcGIS meets the string values in the
  table: Well, Poorly, Somewhat excessively, Excessively, Somewhat poorly) with
  1->3, 2->1, 3->4, 4->5, 5->2, i.e. Excessively 5 > Somewhat excessively 4 > Well 3 >
  Somewhat poorly 2 > Poorly 1 (faster drainage = more vulnerable);
  Reclassify water depth 0-10->5, 10-33->4, 33-61->3, 61-80->2, 80-92->1 (ArcGIS ranges:
  upper bound inclusive);
  vulnerable = 5*drainage + 4*depth, rescaled linearly to [1, 10];
  Reclassify land cover (NLCD table); risk = 8*vulnerable + 10*land_cover;
  high risk = risk > 100; undeveloped = land-cover cells of the 8 undeveloped classes inside
  the high-risk zone (values = NLCD codes), written to pred_results/undeveloped_areas.tif.
"""
import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.features import rasterize
from rasterio.windows import from_bounds, Window

LCC = CRS.from_epsg(32126)  # NAD83 / Oregon North (m)

DRAIN_CODES = ["Well drained", "Poorly drained", "Somewhat excessively drained",
               "Excessively drained", "Somewhat poorly drained"]  # ArcGIS value order (first seen)
DRAIN_RECLASS = {1: 3, 2: 1, 3: 4, 4: 5, 5: 2}
DEPTH_RECLASS = [(0, 10, 5), (10, 33, 4), (33, 61, 3), (61, 80, 2), (80, 92, 1)]
LC_RECLASS = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2,
              81: 5, 82: 9, 90: 1, 95: 1}
UNDEVELOPED = {"Deciduous Forest": 41, "Emergent Herbaceous Wetlands": 95, "Hay/Pasture": 81,
               "Herbaceous": 71, "Mixed Forest": 43, "Shrub/Scrub": 52, "Woody Wetlands": 90,
               "Barren Land": 31}


def reclass_ranges(a, table):
    out = np.full(a.shape, np.nan)
    for i, (lo, hi, v) in enumerate(table):
        sel = (a >= lo) & (a <= hi) if i == 0 else (a > lo) & (a <= hi)
        out[sel & np.isnan(out)] = v
    return out


def main():
    soils = gpd.read_file("dataset/mc_soils.shp").to_crs(LCC)
    boundary = gpd.read_file("dataset/mc_boundary.shp").to_crs(4326).to_crs(LCC)

    with rasterio.open("dataset/land_cover.tif") as src:
        # extent = boundary, snapped to the land_cover grid
        win = from_bounds(*boundary.total_bounds, transform=src.transform)
        win = win.round_offsets(op="floor").round_lengths(op="ceil")
        win = win.intersection(Window(0, 0, src.width, src.height))
        land_cover = src.read(1, window=win)
        transform = src.window_transform(win)
        lc_nodata = src.nodata
    shape = land_cover.shape
    lc_valid = land_cover != lc_nodata

    # Polygon to Raster
    code = {s: i + 1 for i, s in enumerate(DRAIN_CODES)}
    soils["drain_code"] = soils["drclassdcd"].map(code)
    drainage_conditions = rasterize(
        ((g, v) for g, v in zip(soils.geometry, soils["drain_code"]) if v == v),
        out_shape=shape, transform=transform, fill=0, dtype="int16").astype(float)
    drainage_conditions[drainage_conditions == 0] = np.nan
    water_depth = rasterize(
        ((g, v) for g, v in zip(soils.geometry, soils["wdepannmin"]) if v == v),
        out_shape=shape, transform=transform, fill=-9999, dtype="float32").astype(float)
    water_depth[water_depth == -9999] = np.nan

    # Reclassify
    reclass_drainage = np.full(shape, np.nan)
    for k, v in DRAIN_RECLASS.items():
        reclass_drainage[drainage_conditions == k] = v
    reclass_depth = reclass_ranges(water_depth, DEPTH_RECLASS)

    vulnerable_areas = reclass_drainage * 5 + reclass_depth * 4
    vmin, vmax = np.nanmin(vulnerable_areas), np.nanmax(vulnerable_areas)
    reclass_vulnerable = 1 + 9 * (vulnerable_areas - vmin) / (vmax - vmin)

    reclass_land_cover = np.full(shape, np.nan)
    for k, v in LC_RECLASS.items():
        reclass_land_cover[land_cover == k] = v

    risk_zones = reclass_vulnerable * 8 + reclass_land_cover * 10
    high_risk_zones = np.where(risk_zones > 100, risk_zones, np.nan)
    undeveloped = np.isin(land_cover, list(UNDEVELOPED.values())) & np.isfinite(high_risk_zones) & lc_valid
    undeveloped_areas = np.where(undeveloped, land_cover, 0).astype("uint8")

    print(f"vulnerable range {vmin}-{vmax}; risk range {np.nanmin(risk_zones):.1f}-{np.nanmax(risk_zones):.1f}")
    print(f"high-risk cells {np.isfinite(high_risk_zones).sum()}, undeveloped high-risk cells {undeveloped.sum()} "
          f"({undeveloped.sum() * 900 / 1e6:.2f} km2)")
    vals, cnt = np.unique(undeveloped_areas[undeveloped], return_counts=True)
    print("classes:", dict(zip(vals.tolist(), cnt.tolist())))

    os.makedirs("pred_results", exist_ok=True)
    prof = dict(driver="GTiff", height=shape[0], width=shape[1], count=1, crs=LCC, transform=transform,
                compress="lzw")
    with rasterio.open("pred_results/undeveloped_areas.tif", "w", dtype="uint8", nodata=0, **prof) as dst:
        dst.write(undeveloped_areas, 1)
    with rasterio.open("pred_results/risk_zones.tif", "w", dtype="float32", nodata=-9999, **prof) as dst:
        dst.write(np.where(np.isfinite(risk_zones), risk_zones, -9999).astype("float32"), 1)
    return risk_zones, undeveloped_areas


risk_zones, undeveloped_areas = main()
