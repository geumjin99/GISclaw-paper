# Wrong: trusts the EPSG:4326 label of the GeoJSON and reprojects the paths into the DEM
# CRS, then samples speed along the (wrongly placed) lines, clipping indices into the raster.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
src = rasterio.open('dataset/AtlanticDEM.tif'); dem = src.read(1, masked=True).filled(np.nan)
speed = np.sqrt(9.80665 * np.where(dem < 0, -dem, np.nan))
paths = gpd.read_file('dataset/TsuniamiPaths.gpkg')
paths['geometry'] = paths.geometry.clip_by_rect(-180, -90, 180, 90)
paths = paths[~paths.geometry.is_empty]
rows = []
raw = gpd.read_file('dataset/TsuniamiPaths.gpkg')
for _, r in raw.iterrows():
    line = r.geometry.geoms[0]
    lon = np.clip(np.array(line.xy[0]) / 1e5, -180, 180); lat = np.clip(np.array(line.xy[1]) / 1e5, -85, 85)
    g = gpd.GeoSeries(gpd.points_from_xy(np.linspace(lon[0], lon[-1], 500), np.linspace(lat[0], lat[-1], 500)), crs=4326).to_crs(src.crs.to_wkt())
    rr, cc = rasterio.transform.rowcol(src.transform, g.x, g.y)
    rr = np.clip(rr, 0, dem.shape[0] - 1); cc = np.clip(cc, 0, dem.shape[1] - 1)
    sp = speed[rr, cc]; seg = r.lengthkm * 1000 / 499
    rows.append({'Route': r.Route, 'TravelTimeHours': np.nansum(seg / sp) / 3600})
pd.DataFrame(rows).to_csv('pred_results/travel_time.csv', index=False)
plt.imshow(speed); plt.savefig('pred_results/speed.png')
