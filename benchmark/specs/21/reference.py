"""Reference solution for task 21: reclassify NLCD land cover and protected status to a common
1-10 cost scale."""
import os
import numpy as np
import rasterio

landCover_classification = {
    11: 10, 21: 8, 22: 7, 23: 8, 24: 9, 31: 6, 41: 2, 42: 1, 43: 2,
    52: 3, 71: 3, 72: 3, 81: 4, 82: 6, 90: 4, 95: 4, 255: 10,
}
protected_status_classification = {0: 1, 1: 3, 2: 6, 3: 9, 4: 10, 255: 10}


def reclassify(array, table, unmapped=0):
    """Lookup-table reclassification; every output depends only on the ORIGINAL value
    (no chained re-mapping). Values not in the table become `unmapped` (NoData)."""
    lut = np.full(256, unmapped, dtype=np.uint8)
    for k, v in table.items():
        lut[k] = v
    return lut[array]


def run(src_path, table, out_path):
    with rasterio.open(src_path) as src:
        a = src.read(1)
        prof = src.profile
    out = reclassify(a, table)
    unmapped = sorted(set(np.unique(a).tolist()) - set(table))
    print(os.path.basename(src_path), "unmapped input values ->", unmapped)
    prof.update(dtype="uint8", count=1, nodata=0, compress="lzw")
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(out, 1)
    return a, out


os.makedirs("pred_results", exist_ok=True)
landCover, landCover_reclassified = run("dataset/landCover.tif", landCover_classification,
                                        "pred_results/landCover_reclassified.tif")
protected_status, protected_status_reclassified = run("dataset/Protected_Status.tif",
                                                      protected_status_classification,
                                                      "pred_results/protected_status_reclassified.tif")
for name, arr in (("landCover", landCover_reclassified), ("protected", protected_status_reclassified)):
    u, c = np.unique(arr, return_counts=True)
    print(name, dict(zip(u.tolist(), c.tolist())))
