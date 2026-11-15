# Correct analysis (2015 - 2000 change, nodata masked, GeoTIFF written), only the presentation is
# wrong: imshow with origin='lower' draws the map upside down. Fails only the kind="figure"
# orientation check.
import rioxarray, numpy as np, matplotlib.pyplot as plt
a = rioxarray.open_rasterio('dataset/2015_Nature_Africa_PR.2000.tif', masked=True).squeeze()
b = rioxarray.open_rasterio('dataset/2015_Nature_Africa_PR.2015.tif', masked=True).squeeze()
diff = b - a
diff.rio.to_raster('pred_results/malaria_change.tif')
change = diff.values
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(change, cmap='RdBu_r', vmin=-0.6, vmax=0.6, origin='lower')
fig.colorbar(im, label='PfPR change 2000-2015')
fig.savefig('pred_results/malaria_change.png')
