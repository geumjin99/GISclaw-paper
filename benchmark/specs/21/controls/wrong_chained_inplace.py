# Wrong: in-place sequential reclassification; already reclassified values are mapped again
# (0->1, then 1->3, then 3->9 ...).
import rasterio
lc_map = {11:10,21:8,22:7,23:8,24:9,31:6,41:2,42:1,43:2,52:3,71:3,72:3,81:4,82:6,90:4,95:4,255:10}
ps_map = {0:1,1:3,2:6,3:9,4:10,255:10}
for src, mp, dst in [('dataset/landCover.tif', lc_map, 'pred_results/landCover_reclassified.tif'),
                     ('dataset/Protected_Status.tif', ps_map, 'pred_results/protected_status_reclassified.tif')]:
    with rasterio.open(src) as r:
        a = r.read(1); prof = r.profile
    for k, v in mp.items():          # in place: 0->1 then 1->3 then 3->9 ... chains
        a[a == k] = v
    with rasterio.open(dst, 'w', **prof) as w:
        w.write(a, 1)
