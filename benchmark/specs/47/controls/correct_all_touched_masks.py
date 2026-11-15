# Correct: Tabulate Area with all_touched=True masks (one boundary ring of 0.5 ft cells beyond the cell-centre
# rule). A step-by-step session; the steps run in order and those that raise an error are wrapped in try/except.
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)


try:  # this step raises an error

    import geopandas as gpd
    import rasterio
    import numpy as np
    from rasterio.features import geometry_mask
    import json

    # Load parcels
    parcels = gpd.read_file('dataset/parcels.shp')
    print("Parcels columns:", list(parcels.columns))
    print(parcels.head())
    print("CRS:", parcels.crs)
    print("Shape:", parcels.shape)
    print("Bounds:", parcels.total_bounds)

    # Load raster
    with rasterio.open('dataset/impervious.tif') as src:
        print("\nRaster CRS:", src.crs)
        print("Raster size:", src.width, "x", src.height)
        print("Transform:", src.transform)
        print("Res:", src.resolution)
        data = src.read(1)
        print("Value range:", data.min(), data.max())
        vals, counts = np.unique(data, return_counts=True)
        print("Values and counts:", dict(zip(vals.tolist(), counts.tolist())))
        print("nodata:", src.nodata)
        # pixel size
        px = src.res[0] * src.res[1]
        print("Pixel size:", src.res[0], "x", -src.res[1], " area:", px)
except Exception:
    pass


with rasterio.open('dataset/impervious.tif') as src:
    data = src.read(1)
    vals, counts = np.unique(data, return_counts=True)
    print("Values and counts:")
    for v, c in zip(vals, counts):
        print(f"  value={v}  count={c}  area={c * src.res[0] * abs(src.res[1]):.1f} sqft")
    print("Pixel size:", src.res, "nodata:", src.nodata)
    print("Raster bounds:", src.bounds)

print("\nParcels bounds:", parcels.total_bounds)
print("Area_Feet stats:\n", parcels['Area_Feet'].describe())


import geopandas as gpd
vat = gpd.read_file('dataset/impervious.tif.vat.dbf')
print("VAT:")
print(vat)

try:  # this step raises an error
    from rasterio.features import geometry_mask

    parcels = gpd.read_file('dataset/parcels.shp')

    with rasterio.open('dataset/impervious.tif') as src:
        data = src.read(1)
        transform = src.transform
        pixel_area = src.res[0] * abs(src.res[1])  # 0.25 sqft

    tabulate_rows = []
    for fid, row in parcels.iterrows():
        geom = row['geometry']
        mask = geometry_mask(geom, transform=transform, invert=True, all_touched=True)
        # Impervious = value 0 (per VAT)
        n_imp = int(np.sum((data == 0) & mask))
        n_perv = int(np.sum((data == 1) & mask))
        imp_area = n_imp * pixel_area
        perv_area = n_perv * pixel_area
        tabulate_rows.append({
            'FID': fid,
            'Value': 0,
            'Class': 'Impervious',
            'COUNT': n_imp,
            'AREA': round(imp_area, 2),
            'Pervious_COUNT': n_perv,
            'Pervious_AREA': round(perv_area, 2),
        })

    tab = gpd.GeoDataFrame(tabulate_rows, crs=parcels.crs)
    print(tab.head(10))
    print("\nTotals - Impervious px:", tab['COUNT'].sum(), "sqft:", tab['AREA'].sum())
    print("Pervious px:", tab['Pervious_COUNT'].sum(), "sqft:", tab['Pervious_AREA'].sum())
    print("Total parcel area (sqft):", parcels['Area_Feet'].sum())

    # Save tabulate area table (copy of the join result)
    import os
    os.makedirs('pred_results', exist_ok=True)
    tab.to_file('pred_results/Tabulate_Area.shp', driver='ESRI Shapefile')
    tab.to_csv('pred_results/Tabulate_Area.csv', index=False)
    print("Saved Tabulate_Area.shp / .csv")
except Exception:
    pass

try:  # this step raises an error
    from rasterio.features import geometry_mask
    import os

    parcels = gpd.read_file('dataset/parcels.shp')

    with rasterio.open('dataset/impervious.tif') as src:
        data = src.read(1)
        transform = src.transform
        pixel_area = src.res[0] * abs(src.res[1])  # 0.25 sqft

    tabulate_rows = []
    for fid, row in parcels.iterrows():
        geom = row['geometry']
        mask = geometry_mask(geom, out_shape=data.shape, transform=transform, invert=True, all_touched=True)
        n_imp = int(np.sum((data == 0) & mask))
        n_perv = int(np.sum((data == 1) & mask))
        imp_area = n_imp * pixel_area
        perv_area = n_perv * pixel_area
        tabulate_rows.append({
            'FID': fid,
            'Value': 0,
            'Class': 'Impervious',
            'COUNT': n_imp,
            'AREA': round(imp_area, 2),
            'Pervious_COUNT': n_perv,
            'Pervious_AREA': round(perv_area, 2),
        })

    tab = gpd.GeoDataFrame(tabulate_rows, crs=parcels.crs)
    print(tab.head(10))
    print("\nTotals - Impervious px:", tab['COUNT'].sum(), "sqft:", tab['AREA'].sum())
    print("Pervious px:", tab['Pervious_COUNT'].sum(), "sqft:", tab['Pervious_AREA'].sum())
    print("Total parcel area (sqft):", round(parcels['Area_Feet'].sum(), 2))

    os.makedirs('pred_results', exist_ok=True)
    tab.to_file('pred_results/Tabulate_Area.shp', driver='ESRI Shapefile')
    tab.to_csv('pred_results/Tabulate_Area.csv', index=False)
    print("Saved Tabulate_Area.shp / .csv")
except Exception:
    pass

try:  # this step raises an error
    from rasterio.features import rasterize
    import os

    parcels = gpd.read_file('dataset/parcels.shp')

    with rasterio.open('dataset/impervious.tif') as src:
        data = src.read(1)
        transform = src.transform
        pixel_area = src.res[0] * abs(src.res[1])  # 0.25 sqft
        shape = data.shape

    tabulate_rows = []
    for fid, row in parcels.iterrows():
        geom = row['geometry']
        mask = rasterize([(geom, 1)], out_shape=shape, transform=transform, all_touched=True, dtype='uint8').astype(bool)
        n_imp = int(np.sum((data == 0) & mask))
        n_perv = int(np.sum((data == 1) & mask))
        imp_area = n_imp * pixel_area
        perv_area = n_perv * pixel_area
        tabulate_rows.append({
            'FID': fid,
            'Value': 0,
            'Class': 'Impervious',
            'COUNT': n_imp,
            'AREA': round(imp_area, 2),
            'Pervious_COUNT': n_perv,
            'Pervious_AREA': round(perv_area, 2),
        })

    tab = gpd.GeoDataFrame(tabulate_rows, crs=parcels.crs)
    print(tab.head(10))
    print("\nTotals - Impervious px:", tab['COUNT'].sum(), "sqft:", tab['AREA'].sum())
    print("Pervious px:", tab['Pervious_COUNT'].sum(), "sqft:", tab['Pervious_AREA'].sum())
    print("Total parcel area (sqft):", round(parcels['Area_Feet'].sum(), 2))

    os.makedirs('pred_results', exist_ok=True)
    tab.to_file('pred_results/Tabulate_Area.shp', driver='ESRI Shapefile')
    tab.to_csv('pred_results/Tabulate_Area.csv', index=False)
    print("Saved Tabulate_Area.shp / .csv")
except Exception:
    pass

import pandas as pd
import os

tab = pd.DataFrame(tabulate_rows)
print(tab.head(10))
print("\nTotals - Impervious px:", tab['COUNT'].sum(), "sqft:", tab['AREA'].sum())
print("Pervious px:", tab['Pervious_COUNT'].sum(), "sqft:", tab['Pervious_AREA'].sum())
print("Total parcel area (sqft):", round(parcels['Area_Feet'].sum(), 2))

os.makedirs('pred_results', exist_ok=True)
tab.to_csv('pred_results/Tabulate_Area.csv', index=False)

print("Saved Tabulate_Area.csv")
print(tab.describe())

import geopandas as gpd
import pandas as pd
import numpy as np
import os

parcels = gpd.read_file('dataset/parcels.shp')
tab = pd.read_csv('pred_results/Tabulate_Area.csv')

# Add Join: combine tabulate area result with land parcels on FID
joined = parcels.merge(tab, left_on=parcels.index, right_on='FID', how='left')
joined['Impervious_Pct'] = np.round(100 * joined['AREA'] / (joined['AREA'] + joined['Pervious_AREA']), 2)
joined['Pervious_Pct'] = np.round(100 * joined['Pervious_AREA'] / (joined['AREA'] + joined['Pervious_AREA']), 2)

# drop the duplicate FID column
joined = joined.drop(columns=['FID'])

print(joined.head())
print("\nColumns:", list(joined.columns))
print("Rows:", len(joined))
print("\nImpervious area by parcel (sqft):")
print(joined['AREA'].describe())
print("\nOverall impervious pct of classified parcel pixels:",
      round(100 * tab['COUNT'].sum() / (tab['COUNT'].sum() + tab['Pervious_COUNT'].sum()), 2), "%")

os.makedirs('pred_results', exist_ok=True)

# Save the join result as shapefile (copy), attribute table as CSV
joined.to_file('pred_results/Parcels_Impervious_Joined.shp', driver='ESRI Shapefile')
joined.to_csv('pred_results/Parcels_Impervious_Joined.csv', index=False)
print("\nSaved Parcels_Impervious_Joined.shp and .csv in pred_results/")

import geopandas as gpd
import os
import pandas as pd

# Save join result also as GeoPackage (no field name truncation)
joined.to_file('pred_results/Parcels_Impervious_Joined.gpkg', driver='GPKG')

# Verify all outputs
print("Files in pred_results:")
for f in sorted(os.listdir('pred_results')):
    print(f"  {f}  ({os.path.getsize('pred_results/'+f)/1024:.1f} KB)")

# Verify joined shapefile reads back
chk = gpd.read_file('pred_results/Parcels_Impervious_Joined.shp')
print("\nShapefile check:", chk.shape, "CRS ok:", chk.crs.to_epsg() or chk.crs)
print("Cols:", list(chk.columns))

chk2 = gpd.read_file('pred_results/Parcels_Impervious_Joined.gpkg')
print("GPKG cols:", list(chk2.columns))
print(chk2[['Parcel_ID','Area_Feet','COUNT','AREA','Impervious_Pct']].head())

# Sanity: impervious area should be <= total parcel footprint area
print("\nParcels where impervious+pervious > area+5%:",
      ((chk2['AREA']+chk2['Pervious_AREA']) > chk2['Area_Feet']*1.05).sum())
print("\nSummary: total impervious =", chk2['AREA'].sum(), "sqft;",
      "total pervious =", chk2['Pervious_AREA'].sum(), "sqft;",
      "impervious share =", round(100*chk2['AREA'].sum()/(chk2['AREA'].sum()+chk2['Pervious_AREA'].sum()),2), "%")

import geopandas as gpd
chk2 = gpd.read_file('pred_results/Parcels_Impervious_Joined.gpkg')
big = chk2[(chk2['AREA']+chk2['Pervious_AREA']) > chk2['Area_Feet']*1.05]
print(big[['Parcel_ID','Area_Feet','COUNT','AREA','Pervious_COUNT','Pervious_AREA']])
# check if these are near raster edge (raster left=1295796, right=1297529)
for _, r in big.iterrows():
    b = r.geometry.bounds
    print("Parcel", r.Parcel_ID, "bounds:", b, "near raster edge:", b[0] < 1295800 or b[2] > 1297520 or b[1] < 270263 or b[3] > 271868)

