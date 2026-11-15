# Correct: round in WGS84 inside a function, write the de-identified layer back in the source CRS (3857) with counts
import geopandas as gpd, numpy as np
def deidentify(path):
    src = gpd.read_file(path)
    wgs = src.to_crs('EPSG:4326')
    lon = np.round(wgs.geometry.x.to_numpy(), 2); lat = np.round(wgs.geometry.y.to_numpy(), 2)
    g = gpd.GeoDataFrame({'lon2': lon, 'lat2': lat}, geometry=gpd.points_from_xy(lon, lat), crs='EPSG:4326')
    n0 = len(set(zip(wgs.geometry.x, wgs.geometry.y))); n1 = len(set(zip(lon, lat)))
    with open('pred_results/summary.txt', 'w') as f:
        f.write(f'unique locations before {n0}, after {n1}, reduction {n0 - n1} ({100 * (n0 - n1) / n0:.1f}%)\n')
    g.to_crs(src.crs).to_file('pred_results/deidentified_points.gpkg', driver='GPKG')
deidentify('dataset/High_Blood_Level_Results.shp')
