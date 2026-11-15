"""Reference solution for task 26: groundwater vulnerable areas (Morrow County, OR): project, rasterize soils, reclassify,
weighted overlay. Open-source version of the ArcPy ProjectRaster / Project / PolygonToRaster /
Reclassify / Raster Calculator chain."""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, calculate_default_transform

LCC = "EPSG:2269"  # NAD83 / Oregon North (ft): Lambert Conformal Conic (StatePlane FIPS 3601 Feet)


def main():
    soils = gpd.read_file("dataset/mc_soils.shp")          # NAD83 Conus Albers
    boundary = gpd.read_file("dataset/mc_boundary.shp")    # Web Mercator

    # 1. project land cover (nearest neighbour, categorical) and the soils to LCC
    with rasterio.open("dataset/mc_land_cover.tif") as src:
        tr, w, h = calculate_default_transform(src.crs, LCC, src.width, src.height, *src.bounds)
        land_cover = np.full((h, w), 255, dtype="uint8")
        reproject(src.read(1), land_cover, src_transform=src.transform, src_crs=src.crs,
                  dst_transform=tr, dst_crs=LCC, resampling=Resampling.nearest,
                  src_nodata=255, dst_nodata=255)
    soils = soils.to_crs(LCC)
    boundary = boundary.to_crs(LCC)
    cell = tr.a                                             # cellSize = land cover (~98.4 ft)

    # 2. environment: extent = boundary extent, snapped to the projected land-cover grid
    x0, y0 = tr.c, tr.f
    bx0, by0, bx1, by1 = boundary.total_bounds
    c0 = np.floor((bx0 - x0) / cell); c1 = np.ceil((bx1 - x0) / cell)
    r0 = np.floor((y0 - by1) / cell); r1 = np.ceil((y0 - by0) / cell)
    transform = Affine(cell, 0, x0 + c0 * cell, 0, -cell, y0 - r0 * cell)
    shape = (int(r1 - r0), int(c1 - c0))

    # 3. Polygon to Raster (cell centre). drclassdcd is text: ArcGIS codes a string field by
    #    order of first appearance (Well=1, Poorly=2, Somewhat excessively=3, Excessively=4,
    #    Somewhat poorly=5), which is the coding the given remap [[1,3],[2,1],[3,4],[4,5],[5,2]]
    #    refers to (it then ranks drainage ordinally: excessively 5 ... poorly 1).
    classes = list(dict.fromkeys(soils["drclassdcd"].dropna()))
    code = {c: i + 1 for i, c in enumerate(classes)}
    print("drainage codes:", code)
    soils["drain_code"] = soils["drclassdcd"].map(code)
    drainage_conditions = rasterize(zip(soils.geometry, soils["drain_code"]), out_shape=shape,
                                    transform=transform, fill=0, dtype="int16")
    water_depth = rasterize(zip(soils.geometry, soils["wdepannmin"].astype(float)), out_shape=shape,
                            transform=transform, fill=-1, dtype="float32")
    nodata = drainage_conditions == 0

    # 4. Reclassify
    drain_map = {1: 3, 2: 1, 3: 4, 4: 5, 5: 2}
    drain_r = np.zeros(shape, "int16")
    for k, v in drain_map.items():
        drain_r[drainage_conditions == k] = v
    # RemapRange: ArcGIS puts a value on a break into the lower class (0-10 -> 5, (10,33] -> 4 ...)
    depth_r = np.zeros(shape, "int16")
    wd = water_depth
    depth_r[(wd >= 0) & (wd <= 10)] = 5
    depth_r[(wd > 10) & (wd <= 33)] = 4
    depth_r[(wd > 33) & (wd <= 61)] = 3
    depth_r[(wd > 61) & (wd <= 80)] = 2
    depth_r[(wd > 80) & (wd <= 92)] = 1

    # 5. weighted overlay
    vulnerable_areas = (drain_r * 5 + depth_r * 4).astype("int16")
    vulnerable_areas[nodata | (drain_r == 0) | (depth_r == 0)] = -32768
    v = vulnerable_areas[vulnerable_areas != -32768]
    print("vulnerable_areas: cells %d, values %s, mean %.2f" % (v.size, np.unique(v).tolist(), v.mean()))

    prof = {"driver": "GTiff", "height": shape[0], "width": shape[1], "count": 1, "crs": LCC,
            "transform": transform, "compress": "lzw"}
    with rasterio.open("pred_results/vulnerable_areas.tif", "w", dtype="int16", nodata=-32768, **prof) as d:
        d.write(vulnerable_areas, 1)
    with rasterio.open("pred_results/drainage_conditions.tif", "w", dtype="int16", nodata=0, **prof) as d:
        d.write(drainage_conditions, 1)
    with rasterio.open("pred_results/water_depth.tif", "w", dtype="float32", nodata=-1, **prof) as d:
        d.write(water_depth, 1)
    p2 = dict(prof, height=land_cover.shape[0], width=land_cover.shape[1], transform=tr)
    with rasterio.open("pred_results/land_cover.tif", "w", dtype="uint8", nodata=255, **p2) as d:
        d.write(land_cover, 1)


main()
