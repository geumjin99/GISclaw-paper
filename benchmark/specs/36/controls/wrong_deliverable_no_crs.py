"""Wrong: the reference solution of task 36 with every geographic deliverable written without a CRS.

The analysis: hail damage to crop fields, SAVI loss per field (southern Alberta, 3 m imagery).

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
    red = src.read(3, window=win).astype("float32")
    nir = src.read(4, window=win).astype("float32")
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


# ---------------------------------------------------------------------------------------------
# Negative control: the analysis above is the reference solution, but every geographic
# deliverable file is then rewritten without a CRS, which violates the deliverable contract.
def _write_without_crs(path):
    """Rewrite one file in place so that it carries no CRS. Returns a short description.
    .shp: drop the .prj sidecar. .tif/.tiff: rewrite without CRS (and drop .aux.xml, which can
    hold one). .gpkg: rewrite every spatial layer with crs=None. .geojson/.json: write projected
    coordinates (UTM if the data are lon/lat) without a 'crs' member - a lon/lat GeoJSON
    without 'crs' is WGS 84 by definition (RFC 7946), so dropping the member alone would not
    remove the CRS."""
    import os
    low = path.lower()
    stem, _ = os.path.splitext(path)
    if low.endswith(".shp"):
        d, b = os.path.split(stem)
        gone = []
        for n in os.listdir(d or "."):
            if n.lower() == (b + ".prj").lower():
                os.remove(os.path.join(d, n))
                gone.append(n)
        return f"removed {gone}"
    if low.endswith((".tif", ".tiff")):
        import rasterio
        with rasterio.open(path) as r:
            prof, data = r.profile.copy(), r.read()
            tags = r.tags()
        prof.pop("crs", None)
        tmp = stem + ".__nocrs__" + os.path.splitext(path)[1]
        with rasterio.open(tmp, "w", **prof) as w:
            w.write(data)
            if tags:
                w.update_tags(**tags)
        os.replace(tmp, path)
        if os.path.exists(path + ".aux.xml"):
            os.remove(path + ".aux.xml")
        return "raster rewritten without CRS"
    if low.endswith(".gpkg"):
        import geopandas as gpd
        import pyogrio
        tmp = stem + ".__nocrs__.gpkg"
        for lyr, gt in pyogrio.list_layers(path):
            g = gpd.read_file(path, layer=lyr)
            if gt is not None:
                g = g.set_crs(None, allow_override=True)
            g.to_file(tmp, layer=lyr, driver="GPKG")
        os.replace(tmp, path)
        return "gpkg rewritten without CRS"
    if low.endswith((".geojson", ".json")):
        import geopandas as gpd
        g = gpd.read_file(path)
        if g.crs is None or g.crs.is_geographic:
            if g.crs is None:
                g = g.set_crs(4326)
            g = g.to_crs(g.estimate_utm_crs())
        g = g.set_crs(None, allow_override=True)
        os.remove(path)
        g.to_file(path, driver="GeoJSON")
        return "GeoJSON rewritten in projected coordinates without 'crs'"
    return "not a geographic file"


def _strip_deliverable_crs(patterns):
    import fnmatch
    import os
    for n in sorted(os.listdir("pred_results")):
        low = n.lower()
        if low.endswith(('.shp', '.geojson', '.gpkg', '.json', '.tif', '.tiff')) and any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns):
            if low.endswith(".json"):
                try:
                    import json
                    if json.load(open(os.path.join("pred_results", n))).get("type") is None:
                        continue
                except Exception:
                    continue
            print("no-CRS control:", n, _write_without_crs(os.path.join("pred_results", n)))


_strip_deliverable_crs(['*.geojson', '*.gpkg', '*.json', '*.shp'])
