# Correct: the weights are right, but landcover 255 (ocean, unflagged NoData) is summed as cost and drawn
# with the default colour stretch, so the sea takes the top of the scale and the whole land surface is
# one colour. Only the presentation is wrong; the optional readability figure check fails.
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def rd(n):
    with rasterio.open(f"dataset/{n}.tif") as s:
        return s.read(1).astype(float)

distance = rd("distance")
ruggedness = rd("ruggedness")
protected = rd("protected_status_reclassified")
landcover = rd("landcover_reclassified")
cost_surface = 1 * distance + 1.25 * ruggedness + 1 * protected + 1.25 * landcover
plt.imshow(cost_surface, cmap="viridis")
plt.savefig("pred_results/mountainLionCorridor.png", dpi=150, bbox_inches="tight")
