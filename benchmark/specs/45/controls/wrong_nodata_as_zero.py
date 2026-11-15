# Wrong: DEM NoData is set to 0 m and used in the focal window, and the NoData area receives cost values.
import os, numpy as np, rasterio
from scipy.ndimage import maximum_filter, minimum_filter, generic_filter
import matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
src = rasterio.open("dataset/Elevation.tif")
prof = src.profile.copy()
dem = src.read(1).astype("float64")
dem[dem > 1e30] = 0          # nodata -> 0 m, still used in the focal window
rug = maximum_filter(dem, 3) - minimum_filter(dem, 3)
p99 = np.percentile(rug, 99)
cost = 1 + 9 * np.clip(rug / p99, 0, 1)   # robust rescale, nodata area kept as low cost
prof.update(dtype="float32", nodata=-9999, count=1)
with rasterio.open("pred_results/ruggedness_cost.tif", "w", **prof) as d:
    d.write(np.where(np.isfinite(cost), cost, -9999).astype("float32"), 1)
plt.imshow(cost, cmap="magma"); plt.colorbar(); plt.savefig("pred_results/ruggedness_cost.png")
