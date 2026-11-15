# Wrong: adds rounded fields but never rebuilds the points: exact locations remain in the output.
import geopandas as gpd
p = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs('EPSG:4326')
p['LonRound'] = p.geometry.x.round(2); p['LatRound'] = p.geometry.y.round(2)
reduction = len(p) - len(p[['LonRound', 'LatRound']].drop_duplicates())
p.to_file('pred_results/High_Blood_Level_Results_Rounded.shp')
