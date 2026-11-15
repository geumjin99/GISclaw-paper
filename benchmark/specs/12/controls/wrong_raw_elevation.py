# Wrong: the raw elevation is mapped and labelled as ruggedness; no ruggedness index is computed.
import rasterio, matplotlib.pyplot as plt
elev = rasterio.open('dataset/Elevation.tif').read(1)
plt.imshow(elev, cmap='terrain'); plt.colorbar(label='Ruggedness'); plt.title('Ruggedness')
plt.savefig('pred_results/ruggedness.png')
