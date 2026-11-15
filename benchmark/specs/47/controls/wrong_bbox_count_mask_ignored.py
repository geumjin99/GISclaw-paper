# Wrong: rasterio.mask with filled=False, but the mask is ignored when counting -> counts the whole bounding box.
import os, numpy as np, geopandas as gpd, rasterio
from rasterio.mask import mask
os.makedirs("pred_results", exist_ok=True)
parcels = gpd.read_file("dataset/parcels.shp")
rows = []
with rasterio.open("dataset/impervious.tif") as src:
    cell = abs(src.res[0] * src.res[1])
    for pid, geom in zip(parcels.Parcel_ID, parcels.geometry):
        arr, _ = mask(src, [geom], crop=True, filled=False)
        rows.append({"Parcel_ID": pid, "imperv_ft2": float((arr.data[0] == 0).sum() * cell),
                     "perv_ft2": float((arr.data[0] == 1).sum() * cell)})
joined = parcels.merge(gpd.pd.DataFrame(rows), on="Parcel_ID")
joined.to_file("pred_results/parcels_impervious.shp")
