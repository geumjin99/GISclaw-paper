# Wrong: correct rounded layer, but the documented reduction is computed on the un-updated geometry -> "0 fewer locations"
import geopandas as gpd, pandas as pd
pts = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(4326)
pts['x_round'] = pts.geometry.x.round(2); pts['y_round'] = pts.geometry.y.round(2)
new = pts.copy()
new = new.set_geometry(gpd.points_from_xy(new.x_round, new.y_round, crs=4326))
before = len(set(zip(pts.geometry.x, pts.geometry.y)))
after = len(set(zip(pts.geometry.x.round(6), pts.geometry.y.round(6))))   # wrong frame: precision 6, not the rounded points
pd.DataFrame({'unique_before': [before], 'unique_after': [after], 'reduction': [before - after]}).to_csv('pred_results/reduction_report.csv', index=False)
print('Unique locations before', before, 'after', after)
new.to_file('pred_results/rounded_points.shp')
