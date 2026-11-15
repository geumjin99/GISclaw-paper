# Wrong: values reclassified correctly, but the outputs are written without CRS and transform,
# so the georeferencing of the inputs is lost.
import numpy as np, rasterio
lc_map = {11:10,21:8,22:7,23:8,24:9,31:6,41:2,42:1,43:2,52:3,71:3,72:3,81:4,82:6,90:4,95:4,255:10}
ps_map = {0:1,1:3,2:6,3:9,4:10,255:10}
for src, mp, dst in [('dataset/landCover.tif', lc_map, 'pred_results/landCover_reclassified.tif'),
                     ('dataset/Protected_Status.tif', ps_map, 'pred_results/protected_status_reclassified.tif')]:
    a = rasterio.open(src).read(1)
    out = np.vectorize(lambda v: mp.get(int(v), 0))(a).astype('uint8')
    with rasterio.open(dst, 'w', driver='GTiff', height=out.shape[0], width=out.shape[1], count=1,
                       dtype='uint8') as w:     # CRS and transform dropped
        w.write(out, 1)
