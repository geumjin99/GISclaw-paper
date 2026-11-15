# Correct: 3x3 focal range, "rescaling to standardize" read as z-scores of ruggedness.
import os, numpy as np, rasterio
from scipy.ndimage import maximum_filter, minimum_filter, generic_filter
import matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
src = rasterio.open("dataset/Elevation.tif")
prof = src.profile.copy()
dem = src.read(1, masked=True).astype("float64").filled(np.nan)
ok = np.isfinite(dem)
rng = maximum_filter(np.where(ok, dem, -np.inf), 3) - minimum_filter(np.where(ok, dem, np.inf), 3)
rug = np.where(ok, rng, np.nan)
# "rescale to standardise": z-scores of ruggedness
cost = (rug - np.nanmean(rug)) / np.nanstd(rug)
prof.update(dtype="float32", nodata=-9999, count=1)
with rasterio.open("pred_results/ruggedness_cost.tif", "w", **prof) as d:
    d.write(np.where(np.isfinite(cost), cost, -9999).astype("float32"), 1)
plt.imshow(cost, cmap="magma"); plt.colorbar(); plt.savefig("pred_results/ruggedness_cost.png")
