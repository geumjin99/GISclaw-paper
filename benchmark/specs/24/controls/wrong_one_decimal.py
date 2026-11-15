# Wrong: rounds to one decimal instead of two.
import geopandas as gpd
p = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(4326)
p['lon_r'] = p.geometry.x.round(1); p['lat_r'] = p.geometry.y.round(1)
out = gpd.GeoDataFrame(p.drop(columns='geometry'), geometry=gpd.points_from_xy(p.lon_r, p.lat_r), crs=4326)
out.to_file('pred_results/rounded.shp')
