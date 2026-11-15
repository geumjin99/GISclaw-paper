# Correct: shapely/geopandas set_precision on a 0.01 degree grid, de-duplicated locations
# with case counts saved as GeoPackage.
import geopandas as gpd
p = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(4326)
n0 = len(p.geometry.drop_duplicates())
p['geometry'] = p.geometry.set_precision(0.01)
p['x2'] = p.geometry.x; p['y2'] = p.geometry.y
loc = p.groupby(['x2', 'y2']).size().rename('cases').reset_index()
loc = gpd.GeoDataFrame(loc, geometry=gpd.points_from_xy(loc.x2, loc.y2), crs=4326)
n1 = len(loc)
print('unique locations', n0, '->', n1)
loc.to_file('pred_results/deidentified_locations.gpkg')
