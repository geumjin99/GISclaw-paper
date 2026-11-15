# Wrong: stations buffered in DEGREES (0.01 deg ~ 1.1 km N-S but 0.8 km E-W) on the EPSG:4326 layers,
# gap = overlay difference (IoU 0.906 with 1 km metric buffers, because the gap is ~60 % of the
# district); the saved map then shows only the buffers (each .plot without ax= opens a new figure).
import geopandas as gpd, matplotlib.pyplot as plt
from geopandas import overlay

etobicoke = gpd.read_file('dataset/etobicoke.geojson')
fire_stations = gpd.read_file('dataset/fire_station.geojson')
fire_stations_buffered = fire_stations.copy()
fire_stations_buffered['geometry'] = fire_stations_buffered.buffer(0.01)
overlay_result = overlay(etobicoke, fire_stations_buffered, how='difference')
etobicoke.plot(color='white', edgecolor='black')
fire_stations_buffered.plot(color='blue', edgecolor='black')
plt.savefig('pred_results/Fire_Service_Analysis.png', dpi=100, bbox_inches='tight')
plt.close('all')
