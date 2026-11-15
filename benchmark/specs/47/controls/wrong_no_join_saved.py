# Wrong: tabulates correctly but only writes the table as CSV - the join to parcels is never saved.
import geopandas as gpd, pandas as pd
from rasterstats import zonal_stats
p = gpd.read_file('dataset/parcels.shp')
zs = zonal_stats(p, 'dataset/impervious.tif', categorical=True, nodata=255)
pd.DataFrame({'Parcel_ID': p.Parcel_ID, 'imperv_ft2': [z.get(0, 0) * 0.25 for z in zs]}).to_csv('pred_results/impervious_area.csv', index=False)
