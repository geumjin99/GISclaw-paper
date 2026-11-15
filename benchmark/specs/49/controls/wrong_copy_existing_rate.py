# Wrong: the Excel file is not read; the rate field already in the shapefile is reused - no join, no calculation.
import os, geopandas as gpd, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
states = gpd.read_file("dataset/USA_States.shp")
states["homeless_per_10k"] = states["HomelessPe"].astype(float)
states.to_file("pred_results/states_homeless.shp")
ax = states.plot(column="homeless_per_10k", cmap="Reds", legend=True, figsize=(14, 7))
ax.set_xlim(-2.1e7, -7e6)
plt.savefig("pred_results/homeless_rate_2013.png")
