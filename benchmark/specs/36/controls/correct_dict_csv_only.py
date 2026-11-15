# Correct: SAVI loss at 12 m, mean loss per field kept in a dict keyed by field id and written to a CSV only.
import os, shutil, tempfile
import numpy as np, geopandas as gpd, rasterio, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from rasterio.features import rasterize
os.makedirs("pred_results", exist_ok=True)
tmp = tempfile.mkdtemp(); shutil.copy("dataset/Field_boundaries.shp", tmp + "/f.shp")
os.environ["SHAPE_RESTORE_SHX"] = "YES"
fields = gpd.read_file(tmp + "/f.shp").set_crs(3857, allow_override=True)
fields["Field_ID"] = np.arange(1, len(fields) + 1)
sb, sa = rasterio.open("dataset/Before_storm.tif"), rasterio.open("dataset/After_storm.tif")
l = max(sb.bounds.left, sa.bounds.left); r = min(sb.bounds.right, sa.bounds.right)
b = max(sb.bounds.bottom, sa.bounds.bottom); t = min(sb.bounds.top, sa.bounds.top)
F = 4
wb = from_bounds(l, b, r, t, sb.transform).round_offsets().round_lengths()
wa = from_bounds(l, b, r, t, sa.transform).round_offsets().round_lengths()
H, W = int(min(wb.height, wa.height)) // F, int(min(wb.width, wa.width)) // F
def bands(src, win):
    a = src.read([3, 4], window=win, out_shape=(2, H, W)).astype("float32")
    a[a == 0] = np.nan
    return a[0], a[1]
def savi(red, nir, L=0.5):
    red, nir = red / 10000, nir / 10000
    return (nir - red) / (nir + red + L) * (1 + L)
rb, nb = bands(sb, wb); ra, na = bands(sa, wa)
tr = sb.window_transform(wb) * rasterio.Affine.scale(F)
fu = fields.to_crs(sb.crs)
zone = rasterize(zip(fu.geometry, fu.Field_ID), out_shape=(H, W), transform=tr, fill=0, dtype="int32")
loss = savi(rb, nb) - savi(ra, na)
loss[loss < 0] = np.nan
del rb, nb, ra, na
import csv
def summarise(loss, zone):
    damage = {}
    for fid in range(1, len(fields) + 1):
        m = (zone == fid) & np.isfinite(loss)
        if m.any():
            damage[int(fid)] = float(loss[m].mean())
    return damage
damage = summarise(loss, zone)
with open("pred_results/field_damage.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["Field_ID", "mean_savi_loss"])
    for k, val in damage.items():
        wr.writerow([k, val])
plt.figure(figsize=(12, 4)); plt.bar(list(damage.keys()), list(damage.values()))
plt.xlabel("Field_ID"); plt.ylabel("mean SAVI loss")
plt.savefig("pred_results/field_damage.png", dpi=120)
