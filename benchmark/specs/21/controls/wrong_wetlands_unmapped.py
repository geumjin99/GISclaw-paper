# Wrong: classes 41, 90 and 95 are missing from the table -> those cells keep raw codes 41/90/95 (off the 1-10 scale)
import numpy as np, rasterio
LC = {11: 10, 21: 8, 22: 7, 23: 8, 24: 9, 31: 6, 42: 1, 43: 2, 52: 3, 71: 3, 72: 3, 81: 4, 82: 6, 255: 10}
PS = {0: 1, 1: 3, 2: 6, 3: 9, 4: 10, 255: 10}
for src, tab, dst in (("dataset/landCover.tif", LC, "pred_results/landCover_reclassified.tif"),
                      ("dataset/Protected_Status.tif", PS, "pred_results/protected_status_reclassified.tif")):
    with rasterio.open(src) as s:
        a = s.read(1); prof = s.profile
    out = a.copy()
    for k, v in tab.items():
        out[a == k] = v
    with rasterio.open(dst, "w", **prof) as d:
        d.write(out, 1)
