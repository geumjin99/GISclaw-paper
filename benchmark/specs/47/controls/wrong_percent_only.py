# Wrong: saves only the impervious percentage (share), not the tabulated area.
import geopandas as gpd
from rasterstats import zonal_stats
p = gpd.read_file('dataset/parcels.shp')
zs = zonal_stats(p, 'dataset/impervious.tif', categorical=True, nodata=255)
p['imp_pct'] = [100 * z.get(0, 0) / max(1, z.get(0, 0) + z.get(1, 0)) for z in zs]
p.to_file('pred_results/parcels_joint.shp')
