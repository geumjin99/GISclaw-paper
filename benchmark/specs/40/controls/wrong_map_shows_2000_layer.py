# Wrong: computes the difference correctly, but the saved map shows the 2000 prevalence surface, not the change.
import os, numpy as np, rasterio, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
a = rasterio.open("dataset/2015_Nature_Africa_PR.2000.tif"); b = rasterio.open("dataset/2015_Nature_Africa_PR.2015.tif")
pr00 = a.read(1, masked=True); pr15 = b.read(1, masked=True)
diff = pr15 - pr00
prof = a.profile; prof.update(dtype="float32", nodata=-9999)
with rasterio.open("pred_results/malaria_change.tif", "w", **prof) as d:
    d.write(diff.filled(-9999).astype("float32"), 1)
fig, ax = plt.subplots(figsize=(9, 9))
im = ax.imshow(pr00, cmap="RdBu_r", vmin=-1, vmax=1)
fig.colorbar(im, label="PfPR change"); ax.set_title("Malaria prevalence change 2000-2015")
fig.savefig("pred_results/malaria_change.png")
