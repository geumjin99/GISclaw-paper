# Wrong: the raster CRS says "metre" (LOCAL_CS) -> cell area taken as 0.25 m2, then "converted" to ft2 (x10.764): values 10.8x too large.
import os, numpy as np, geopandas as gpd, pandas as pd
from rasterstats import zonal_stats
os.makedirs("pred_results", exist_ok=True)
parcels = gpd.read_file("dataset/parcels.shp")
zs = zonal_stats(parcels, "dataset/impervious.tif", categorical=True, nodata=255)
cell_m2 = 0.5 * 0.5
tab = pd.DataFrame({"Parcel_ID": parcels.Parcel_ID, "Imperv_m2": [z.get(0, 0) * cell_m2 for z in zs]})
tab["Imperv_ft2"] = tab.Imperv_m2 * 10.7639
parcels.merge(tab, on="Parcel_ID").to_file("pred_results/parcels_impervious.geojson", driver="GeoJSON")
