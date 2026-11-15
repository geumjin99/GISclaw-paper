# Wrong: table transcription error; open water 11 -> 1 (lowest cost) instead of 10 (barrier); everything else right
import numpy as np, rasterio, os
LC = {11: 1, 21: 8, 22: 7, 23: 8, 24: 9, 31: 6, 41: 2, 42: 1, 43: 2, 52: 3, 71: 3, 72: 3, 81: 4, 82: 6, 90: 4, 95: 4, 255: 10}
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
