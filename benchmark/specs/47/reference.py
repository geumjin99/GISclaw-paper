"""Reference solution for task 47: impervious area per land parcel (Tabulate Area + join + copy).

impervious.tif: 0.5 ft cells (NAD83 / Kentucky North, US ft), classes from its attribute
table (impervious.tif.vat.dbf): Value 0 = Impervious, Value 1 = Pervious, 255 = NoData.
Tabulate Area (cell centres within each parcel) -> area per class in square feet
(cell count x 0.25 ft2), fields VALUE_0 / VALUE_1 as ArcGIS names them, joined to the
121 parcels on Parcel_ID and copied to pred_results/parcels_joint.shp.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats

parcels = gpd.read_file("dataset/parcels.shp")
with rasterio.open("dataset/impervious.tif") as src:
    cell_area = abs(src.res[0] * src.res[1])        # 0.25 ft2
    nodata = src.nodata
print("parcels:", len(parcels), "cell area ft2:", cell_area)

zs = zonal_stats(parcels, "dataset/impervious.tif", categorical=True, nodata=nodata, all_touched=False)
tab = pd.DataFrame([{"Parcel_ID": pid, "VALUE_0": z.get(0, 0) * cell_area, "VALUE_1": z.get(1, 0) * cell_area}
                    for pid, z in zip(parcels["Parcel_ID"], zs)])
impervious_area = tab                                 # Tabulate Area result (one row per parcel)

parcels_joint = parcels.merge(impervious_area, on="Parcel_ID", how="left")   # Add Join (KEEP_ALL)
parcels_joint["Imperv_ft2"] = parcels_joint["VALUE_0"]
parcels_joint["Imperv_pct"] = 100 * parcels_joint["VALUE_0"] / (parcels_joint["VALUE_0"] + parcels_joint["VALUE_1"])
parcels_joint.to_file("pred_results/parcels_joint.shp")             # Copy Features
print("total impervious ft2: %.0f, pervious ft2: %.0f, mean impervious share %.1f %%" % (
    parcels_joint.VALUE_0.sum(), parcels_joint.VALUE_1.sum(), parcels_joint.Imperv_pct.mean()))
