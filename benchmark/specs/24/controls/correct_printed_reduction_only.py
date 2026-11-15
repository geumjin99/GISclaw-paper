# Correct: rounded layer (geometry only); the reduction is computed and printed, not written to a file.
import geopandas as gpd
from shapely.geometry import Point
pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(epsg=4326)
pts['x_rounded'] = pts.geometry.x.round(2)
pts['y_rounded'] = pts.geometry.y.round(2)
gdf_rounded = gpd.GeoDataFrame(geometry=pts.apply(lambda r: Point(r['x_rounded'], r['y_rounded']), axis=1), crs=pts.crs)
print('Number of unique locations before rounding:', pts.geometry.unique().size)
print('Number of unique locations after rounding:', gdf_rounded.geometry.unique().size)
gdf_rounded.to_file('pred_results/High_Blood_Level_Results_Rounded.shp')
