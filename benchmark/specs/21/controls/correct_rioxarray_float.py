# Correct: rioxarray, float32 output, NaN for values outside the table
import numpy as np, rioxarray
def reclass(path, table, out):
    da = rioxarray.open_rasterio(path).squeeze('band', drop=True)
    vals = da.values
    res = np.full(vals.shape, np.nan, dtype='float32')
    for k, v in table.items():
        res[vals == k] = v
    o = da.copy(data=res).rio.write_nodata(np.nan)
    o.rio.to_raster(out)
reclass('dataset/landCover.tif', {11: 10, 21: 8, 22: 7, 23: 8, 24: 9, 31: 6, 41: 2, 42: 1, 43: 2, 52: 3, 71: 3, 72: 3,
                                  81: 4, 82: 6, 90: 4, 95: 4, 255: 10}, 'pred_results/landCover_reclassified.tif')
reclass('dataset/Protected_Status.tif', {0: 1, 1: 3, 2: 6, 3: 9, 4: 10, 255: 10}, 'pred_results/protected_status_reclassified.tif')
