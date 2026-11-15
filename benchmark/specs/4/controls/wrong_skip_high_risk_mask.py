# Wrong: selects the undeveloped land-cover classes but never applies the high-risk mask.
import numpy as np, rasterio
with rasterio.open('dataset/land_cover.tif') as src:
    lc = src.read(1); prof = src.profile
out = np.where(np.isin(lc, [41, 95, 81, 71, 43, 52, 90, 31]), lc, 0).astype('uint8')
prof.update(nodata=0, crs='EPSG:32126')
with rasterio.open('pred_results/undeveloped_areas.tif', 'w', **prof) as d:
    d.write(out, 1)
