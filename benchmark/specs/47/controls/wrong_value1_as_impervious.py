# Wrong: assumes 1 = impervious (it is 'Pervious' in the VAT); zonal sum of the 1-cells.
import geopandas as gpd
from rasterstats import zonal_stats
p = gpd.read_file('dataset/parcels.shp')
zs = zonal_stats(p, 'dataset/impervious.tif', stats=['sum', 'count'], nodata=255)
p['imperv_ft2'] = [z['sum'] * 0.25 for z in zs]
p.to_file('pred_results/parcels_joint.shp')
