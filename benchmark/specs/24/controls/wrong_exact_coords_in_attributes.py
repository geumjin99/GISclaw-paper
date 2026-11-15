# Wrong: rounded geometry, but the exact lon/lat are kept as attribute fields in the published layer
import geopandas as gpd, pandas as pd
pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(4326)
pts['orig_lon'] = pts.geometry.x; pts['orig_lat'] = pts.geometry.y
pts['lon_r'] = pts.orig_lon.round(2); pts['lat_r'] = pts.orig_lat.round(2)
out = gpd.GeoDataFrame(pts.drop(columns='geometry'), geometry=gpd.points_from_xy(pts.lon_r, pts.lat_r), crs=4326)
before = pts[['orig_lon', 'orig_lat']].drop_duplicates().shape[0]
after = out[['lon_r', 'lat_r']].drop_duplicates().shape[0]
print('unique locations', before, '->', after, 'reduction', before - after)
out.to_file('pred_results/blood_lead_rounded.shp')
