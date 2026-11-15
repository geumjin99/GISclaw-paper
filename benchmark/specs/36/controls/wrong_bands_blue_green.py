"""Wrong: bands 1 and 2 (blue, green) are taken as red and NIR, so the index is not SAVI.

The remaining steps follow the reference solution: hail damage to crop fields, SAVI loss per field
(southern Alberta, 3 m imagery).

Inputs: Before_storm.tif (9030 x 9121) and After_storm.tif (7293 x 8549), 4-band uint16
(B, G, R, NIR; nodata 0) on the same 3 m UTM 12N lattice but with different extents;
Field_boundaries.shp: 319 polygons, shapefile without .shx/.dbf/.prj - the coordinates are
Web Mercator (EPSG:3857), no Field_ID attribute (row order is used as Field_ID).

SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L), L = 0.5, on reflectance (DN / 10000);
vegetation loss = SAVI_before - SAVI_after on the overlap of the two images; negative values
(gain) remapped to NoData as in the authors' model; mean loss per field (zonal statistics,
cell-centre rule); joined to the fields and mapped. Memory: processed in row strips.
"""
import os
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds, Window

os.makedirs("pred_results", exist_ok=True)
L = 0.5

# fields: restore the missing .shx in a writable temp dir; CRS = Web Mercator
tmp = tempfile.mkdtemp()
shutil.copy("dataset/Field_boundaries.shp", os.path.join(tmp, "fields.shp"))
os.environ["SHAPE_RESTORE_SHX"] = "YES"
fields = gpd.read_file(os.path.join(tmp, "fields.shp"))
fields = fields.set_crs(3857, allow_override=True)
fields["Field_ID"] = np.arange(1, len(fields) + 1)

src_b = rasterio.open("dataset/Before_storm.tif")
src_a = rasterio.open("dataset/After_storm.tif")
fields_utm = fields.to_crs(src_b.crs)

# overlap window on each image (same 3 m lattice)
left = max(src_b.bounds.left, src_a.bounds.left); right = min(src_b.bounds.right, src_a.bounds.right)
bottom = max(src_b.bounds.bottom, src_a.bounds.bottom); top = min(src_b.bounds.top, src_a.bounds.top)
wb = from_bounds(left, bottom, right, top, src_b.transform).round_offsets().round_lengths()
wa = from_bounds(left, bottom, right, top, src_a.transform).round_offsets().round_lengths()
H, W = int(min(wb.height, wa.height)), int(min(wb.width, wa.width))
tr = src_b.window_transform(wb)
zone = rasterize(((g, i) for g, i in zip(fields_utm.geometry, fields_utm["Field_ID"])), out_shape=(H, W),
                 transform=tr, fill=0, dtype="int32")


def savi(src, win):
    red = src.read(1, window=win).astype("float32")
    nir = src.read(2, window=win).astype("float32")
    ok = (red > 0) & (nir > 0)
    red /= 10000.0; nir /= 10000.0
    out = (nir - red) / (nir + red + L) * (1 + L)
    out[~ok] = np.nan
    return out


def strips():
    n = len(fields) + 1
    s_sum = np.zeros(n); s_cnt = np.zeros(n); raw_sum = np.zeros(n); raw_cnt = np.zeros(n)
    preview = []
    step = 512
    for r0 in range(0, H, step):
        h = min(step, H - r0)
        loss = savi(src_b, Window(wb.col_off, wb.row_off + r0, W, h)) - \
            savi(src_a, Window(wa.col_off, wa.row_off + r0, W, h))
        z = zone[r0:r0 + h]
        v = np.isfinite(loss) & (z > 0)
        raw_sum += np.bincount(z[v], loss[v], n); raw_cnt += np.bincount(z[v], minlength=n)
        rem = v & (loss >= 0)                      # remap: negative loss (gain) -> NoData
        s_sum += np.bincount(z[rem], loss[rem], n); s_cnt += np.bincount(z[rem], minlength=n)
        preview.append(loss[::10, ::10])
    return s_sum, s_cnt, raw_sum, raw_cnt, np.vstack(preview)


s_sum, s_cnt, raw_sum, raw_cnt, loss_preview = strips()
with np.errstate(invalid="ignore", divide="ignore"):
    mean_loss = s_sum / s_cnt
    mean_change = raw_sum / raw_cnt
fields["MEAN_veg_loss"] = mean_loss[fields["Field_ID"].values]          # remapped (loss >= 0 only)
fields["MEAN_savi_change"] = mean_change[fields["Field_ID"].values]     # all pixels, before - after
fields["n_pixels"] = raw_cnt[fields["Field_ID"].values]
del zone  # 250 MB label raster, not needed further
field_loss = fields.to_crs(src_b.crs)
print("fields with data:", int(field_loss["MEAN_veg_loss"].notna().sum()), "of", len(field_loss),
      "| mean loss:", float(field_loss["MEAN_veg_loss"].mean()))
field_loss.to_file("pred_results/field_vegetation_loss.geojson", driver="GeoJSON")
field_loss.drop(columns="geometry").to_csv("pred_results/field_vegetation_loss.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
ext = [tr.c, tr.c + tr.a * W, tr.f + tr.e * H, tr.f]
im = axes[0].imshow(loss_preview, cmap="RdYlGn_r", vmin=-0.3, vmax=0.3, extent=ext)
fig.colorbar(im, ax=axes[0], shrink=0.7, label="SAVI before - after")
field_loss.boundary.plot(ax=axes[0], color="k", linewidth=0.3)
axes[0].set_title("Pixel-level vegetation loss (SAVI)")
field_loss.plot(ax=axes[1], column="MEAN_veg_loss", cmap="OrRd", legend=True, edgecolor="grey", linewidth=0.3,
                missing_kwds={"color": "lightgrey"}, legend_kwds={"shrink": 0.7, "label": "mean SAVI loss per field"})
axes[1].set_title("Mean vegetation loss per field (hail damage)")
for a in axes:
    a.set_axis_off()
plt.savefig("pred_results/hail_damage_fields.png", dpi=150, bbox_inches="tight")
