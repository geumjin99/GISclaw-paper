# Correct: per-parcel rasterio.mask, impervious = class 0 (from the VAT), area in square metres, GeoPackage.
import numpy as np, geopandas as gpd, rasterio
from rasterio.mask import mask
vat = gpd.read_file('dataset/impervious.tif.vat.dbf')
imp_val = int(vat.loc[vat.Class_name == 'Impervious', 'Value'].iloc[0])
p = gpd.read_file('dataset/parcels.shp')
res = []
with rasterio.open('dataset/impervious.tif') as src:
    ca = abs(src.res[0] * src.res[1]) * 0.3048006096 ** 2
    for geom in p.geometry:
        a, _ = mask(src, [geom], crop=True, filled=True, nodata=255)
        res.append((a[0] == imp_val).sum() * ca)
p['imp_m2'] = res
p.to_file('pred_results/parcels_impervious.gpkg', driver='GPKG')
