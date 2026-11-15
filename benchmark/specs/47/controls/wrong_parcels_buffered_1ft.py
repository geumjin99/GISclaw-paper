# Wrong: impervious = class 0 from the VAT, but tabulated inside parcels buffered outward by 1 ft
# (2 cells: adjacent sidewalk / neighbours counted), about +2.6 % in total.
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import geometry_mask
vat = gpd.read_file('dataset/impervious.tif.vat.dbf')
imp_val = int(vat.loc[vat.Class_name == 'Impervious', 'Value'].iloc[0])
p = gpd.read_file('dataset/parcels.shp')
with rasterio.open('dataset/impervious.tif') as src:
    data = src.read(1); T = src.transform; ca = abs(src.res[0] * src.res[1])
p['imp_sqft'] = [float(np.sum((data == imp_val) & geometry_mask([g.buffer(1)], out_shape=data.shape, transform=T,
                                                                   invert=True))) * ca for g in p.geometry]
p.to_file('pred_results/parcels_impervious.shp')
