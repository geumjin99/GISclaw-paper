# Wrong: reversed subtraction (2000 - 2015) drawn with RdBu (red = low values); neither code nor
# figure states the subtraction order.
import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
os.makedirs('pred_results', exist_ok=True)
a = rasterio.open('dataset/2015_Nature_Africa_PR.2000.tif').read(1, masked=True)
b = rasterio.open('dataset/2015_Nature_Africa_PR.2015.tif').read(1, masked=True)
change = (a - b).filled(np.nan)
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(np.ma.masked_invalid(change), cmap='RdBu', norm=TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=0.6))
fig.colorbar(im, ax=ax, label='Change in PfPR2-10')
ax.set_title('Malaria prevalence change, Sub-Saharan Africa')
fig.savefig('pred_results/malaria_change.png', dpi=120)
