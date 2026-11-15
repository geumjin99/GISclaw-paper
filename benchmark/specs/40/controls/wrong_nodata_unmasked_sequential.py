# Wrong: nodata (-9999) not masked -> ocean shown as zero change; sequential colour map (no increase/decrease split).
import os, numpy as np, rasterio, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
d00 = rasterio.open("dataset/2015_Nature_Africa_PR.2000.tif").read(1)
d15 = rasterio.open("dataset/2015_Nature_Africa_PR.2015.tif").read(1)
change = d15 - d00
plt.figure(figsize=(9, 9))
plt.imshow(change, cmap="viridis", vmin=-0.6, vmax=0.2); plt.colorbar(label="change in PfPR")
plt.savefig("pred_results/malaria_change.png")
