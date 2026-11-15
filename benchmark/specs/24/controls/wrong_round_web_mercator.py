# Wrong: CRS mistake, rounds the Web Mercator metre coordinates (no reprojection to WGS 84).
import geopandas as gpd
from shapely.geometry import Point
p = gpd.read_file('dataset/High_Blood_Level_Results.shp')
p['X_r'] = p.geometry.x.round(2); p['Y_r'] = p.geometry.y.round(2)
p['geometry'] = [Point(a, b) for a, b in zip(p.X_r, p.Y_r)]
before = len(p); after = len(p[['X_r', 'Y_r']].drop_duplicates())
print(before, after)
p.to_file('pred_results/High_Blood_Level_Results_Rounded.shp')
