# Wrong: a 9x9 focal window instead of the 3x3 window the instruction asks for.
import os, numpy as np, rasterio
from scipy.ndimage import maximum_filter, minimum_filter, generic_filter
import matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
src = rasterio.open("dataset/Elevation.tif")
prof = src.profile.copy()
dem = src.read(1, masked=True).astype("float64").filled(np.nan)
ok = np.isfinite(dem)
rng = maximum_filter(np.where(ok, dem, -np.inf), 9) - minimum_filter(np.where(ok, dem, np.inf), 9)   # 9x9 instead of 3x3
rug = np.where(ok, rng, np.nan)
cost = 1 + 9 * (rug - np.nanmin(rug)) / (np.nanmax(rug) - np.nanmin(rug))
prof.update(dtype="float32", nodata=-9999, count=1)
with rasterio.open("pred_results/ruggedness_cost.tif", "w", **prof) as d:
    d.write(np.where(np.isfinite(cost), cost, -9999).astype("float32"), 1)
plt.imshow(cost, cmap="magma"); plt.colorbar(); plt.savefig("pred_results/ruggedness_cost.png")
