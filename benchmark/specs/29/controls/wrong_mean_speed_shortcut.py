# Wrong: skips the along-path analysis - route length (lengthkm) divided by the mean ocean speed.
import numpy as np, pandas as pd, geopandas as gpd, rasterio, matplotlib.pyplot as plt
dem = rasterio.open('dataset/AtlanticDEM.tif').read(1, masked=True).filled(np.nan)
speed = np.sqrt(9.80665 * -dem[dem < 0])
paths = gpd.read_file('dataset/TsuniamiPaths.gpkg')
paths['TravelTimeHours'] = paths.lengthkm * 1000 / np.nanmean(speed) / 3600
paths[['Route', 'TravelTimeHours']].to_csv('pred_results/travel_time.csv', index=False)
plt.imshow(np.where(dem < 0, np.sqrt(9.80665 * -np.where(dem < 0, dem, 0)), np.nan)); plt.savefig('pred_results/speed.png')
