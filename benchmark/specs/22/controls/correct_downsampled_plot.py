# Correct: weighted sum (unnormalised), NoData masked, map drawn from a 4x decimated array.
import numpy as np, rasterio, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def rd(n):
    with rasterio.open(f"dataset/{n}.tif") as s:
        return s.read(1).astype("float32")
d, r, p, l = (rd(n) for n in ["distance", "ruggedness", "protected_status_reclassified", "landcover_reclassified"])
l[l == 255] = np.nan
cost = d + 1.25 * r + p + 1.25 * l
plt.figure(figsize=(9, 7)); plt.imshow(cost[::4, ::4], cmap="inferno"); plt.colorbar(label="cost")
plt.title("Mountain lion corridor cost"); plt.savefig("pred_results/mountainLionCorridor.png", dpi=100)
