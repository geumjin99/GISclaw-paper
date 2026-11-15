# Correct: vectorize, dissolve into one open-space feature, store area (m2, km2, acres) on it
import numpy as np, rasterio, geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape
with rasterio.open('dataset/PADUS_CRS_clip_1.tif') as s:
    a = s.read(1); prof = s.profile; nd = s.nodata; T = s.transform; crs = s.crs
os_ = np.where((a != nd) & (a > 0), 1, 0).astype('uint8')
prof.update(dtype='uint8', nodata=255)
with rasterio.open('pred_results/open_space.tif', 'w', **prof) as d:
    d.write(os_, 1)
geoms = [shape(p) for p, v in shapes(os_, mask=os_ == 1, transform=T)]
parts = gpd.GeoDataFrame({'value': [1] * len(geoms)}, geometry=geoms, crs=crs)
dis = parts.dissolve(by='value').reset_index()
dis['area_m2'] = dis.geometry.area; dis['area_km2'] = dis['area_m2'] / 1e6; dis['acres'] = dis['area_m2'] * 0.000247105
dis['n_parts'] = len(parts)
dis.to_file('pred_results/open_space.shp')
print(dis[['area_m2', 'area_km2', 'acres']])
