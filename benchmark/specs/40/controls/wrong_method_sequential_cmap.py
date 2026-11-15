# Wrong: the task prescribes a diverging color scheme; this is correct_func_pp_xarray drawn with a
# sequential colormap (viridis, linear vmin/vmax).
import numpy as np, xarray as xr, rioxarray, matplotlib.pyplot as plt

def main():
    p0 = rioxarray.open_rasterio("dataset/2015_Nature_Africa_PR.2000.tif", masked=True).squeeze("band", drop=True)
    p1 = rioxarray.open_rasterio("dataset/2015_Nature_Africa_PR.2015.tif", masked=True).squeeze("band", drop=True)
    delta_pp = (p1 - p0) * 100.0
    fig, ax = plt.subplots(figsize=(10, 10))
    delta_pp.plot.imshow(ax=ax, cmap="viridis", vmin=-60, vmax=20,
                         cbar_kwargs={"label": "PfPR2-10 change 2000-2015 (percentage points)"})
    ax.set_title("Change in malaria prevalence 2000-2015")
    fig.savefig("pred_results/pfpr_change_2000_2015.png", dpi=110)

main()
