# Correct: xarray-based Minus in percentage points, diverging map, everything in a function, no GeoTIFF
import numpy as np, xarray as xr, rioxarray, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

def main():
    p0 = rioxarray.open_rasterio("dataset/2015_Nature_Africa_PR.2000.tif", masked=True).squeeze("band", drop=True)
    p1 = rioxarray.open_rasterio("dataset/2015_Nature_Africa_PR.2015.tif", masked=True).squeeze("band", drop=True)
    delta_pp = (p1 - p0) * 100.0
    fig, ax = plt.subplots(figsize=(10, 10))
    delta_pp.plot.imshow(ax=ax, cmap="PuOr_r", norm=TwoSlopeNorm(0, -60, 20),
                         cbar_kwargs={"label": "PfPR2-10 change 2000-2015 (percentage points)"})
    ax.set_title("Increase (orange) vs decrease (purple)")
    fig.savefig("pred_results/pfpr_change_2000_2015.png", dpi=110)

main()
