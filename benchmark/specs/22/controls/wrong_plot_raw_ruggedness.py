# Wrong: plots an input layer instead of the composite.
import rasterio, matplotlib.pyplot as plt
rug = rasterio.open('dataset/ruggedness.tif').read(1)
plt.imshow(rug, cmap='viridis'); plt.colorbar(); plt.title('cost'); plt.savefig('pred_results/mountainLionCorridor.png')
