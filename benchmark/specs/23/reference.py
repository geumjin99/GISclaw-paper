"""Reference solution for task 23: open space (PAD-US) for CRS flood-insurance credit - reclassify,
vectorize, area."""
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

SQM_TO_ACRES = 0.000247105
IMPERVIOUS_ACCURACY = 0.85  # NLCD impervious-surface accuracy used in the CRS lesson


def main():
    with rasterio.open("dataset/PADUS_CRS_clip_1.tif") as src:
        padus = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata                 # 2147483648 (uint32): outside any PAD-US unit
        transform, crs = src.transform, src.crs
    # The raster is thematic: each cell holds the OBJECTID of a PAD-US protected area
    # (eligible open space). Threshold: a valid PAD-US id (> 0 and != NoData) -> open space.
    valid = (padus != nodata) & (padus > 0)
    open_space = valid.astype("uint8")
    cell_area = abs(transform.a * transform.e)              # 900 m2
    print("open-space cells:", int(open_space.sum()), "of", open_space.size)

    profile.update(dtype="uint8", nodata=255, count=1, compress="lzw")
    with rasterio.open("pred_results/open_space.tif", "w", **profile) as dst:
        dst.write(open_space, 1)

    polys = [shape(geom) for geom, val in shapes(open_space, mask=open_space == 1,
                                                  transform=transform, connectivity=4) if val == 1]
    gdf = gpd.GeoDataFrame({"gridcode": np.ones(len(polys), dtype=int)}, geometry=polys, crs=crs)
    gdf["area_sqm"] = gdf.geometry.area
    gdf["acres"] = gdf["area_sqm"] * SQM_TO_ACRES
    gdf["acres_adj"] = gdf["acres"] * IMPERVIOUS_ACCURACY   # CRS-lesson credited acres
    gdf.to_file("pred_results/open_space.shp")

    total_sqm = float(gdf["area_sqm"].sum())
    assert abs(total_sqm - open_space.sum() * cell_area) < 1.0
    total_acres = total_sqm * SQM_TO_ACRES
    print(f"Open space polygons: {len(gdf)}")
    print(f"Total open space area: {total_sqm:,.0f} m2 = {total_sqm / 1e6:.2f} km2 = {total_acres:,.1f} acres"
          f" ({total_acres * IMPERVIOUS_ACCURACY:,.1f} acres after the 0.85 accuracy factor)")


main()
