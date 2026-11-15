"""Checks for task 21: reclassify land cover and protected status to a common 1-10 scale.

The mapping is given explicitly in the task (Domain Knowledge), so correctness is exact:
Required, for each of the two outputs:
  1. the file exists (instructed name; .tif/.tiff and case variants accepted);
  2. georeferencing is preserved: same CRS and same grid (shape + transform within 1 m)
     as the input raster, so the layers stay overlayable for the later cost-surface step;
  3. >= 99.5 % of the cells whose input value is a key of the given table carry exactly the
     tabulated output value. Cells whose input value is not in the table (land cover 0,
     2.1 % of cells, outside the NLCD footprint) are ignored, so any NoData convention is
     accepted. The same >= 99.5 % must hold within EVERY tabulated input class present, so
     one class mapped wrongly or forgotten (e.g. open water, 0.45 % of cells) fails. A chained (in-place, sequential) reclassification, swapped tables or treating
     255 as NoData instead of class 10 all fail here.
"""
import glob
import os

import numpy as np

from checklib import result

LC = {11: 10, 21: 8, 22: 7, 23: 8, 24: 9, 31: 6, 41: 2, 42: 1, 43: 2,
      52: 3, 71: 3, 72: 3, 81: 4, 82: 6, 90: 4, 95: 4, 255: 10}
PS = {0: 1, 1: 3, 2: 6, 3: 9, 4: 10, 255: 10}
OUTPUTS = [("landCover_reclassified", "landCover.tif", LC),
           ("protected_status_reclassified", "Protected_Status.tif", PS)]


def _find(ctx, stem):
    for p in glob.glob(os.path.join(ctx.pred_dir, "*")):
        b = os.path.basename(p)
        root, ext = os.path.splitext(b)
        if root.lower() == stem.lower() and ext.lower() in (".tif", ".tiff"):
            return p
    return None


def checks(ctx):
    import rasterio
    out = []
    for stem, src, table in OUTPUTS:
        p = _find(ctx, stem)
        out.append(result(f"{stem}: file present", p is not None, value=os.path.basename(p) if p else None,
                          threshold=[stem + ".tif", stem + ".tiff"]))
        if p is None:
            out.append(result(f"{stem}: georeferencing preserved", False, note="no file"))
            out.append(result(f"{stem}: tabulated values (>=99.5 % of mapped cells, and of every class)", False, note="no file"))
            continue
        try:
            with rasterio.open(os.path.join(ctx.inputs_dir, src)) as r:
                a_in, crs_in, tr_in = r.read(1), r.crs, r.transform
            with rasterio.open(p) as r:
                a_out, crs_out, tr_out = r.read(1), r.crs, r.transform
        except Exception as e:
            out.append(result(f"{stem}: georeferencing preserved", False, note=repr(e)[:200]))
            out.append(result(f"{stem}: tabulated values (>=99.5 % of mapped cells, and of every class)", False, note="unreadable"))
            continue
        same_grid = (a_out.shape == a_in.shape and crs_out is not None and crs_in == crs_out
                     and all(abs(x - y) <= 1.0 for x, y in zip(tuple(tr_in)[:6], tuple(tr_out)[:6])))
        out.append(result(f"{stem}: georeferencing preserved", same_grid,
                          value={"shape": list(a_out.shape), "crs": str(crs_out)[:40], "origin": [tr_out.c, tr_out.f]},
                          threshold={"shape": list(a_in.shape), "origin": [tr_in.c, tr_in.f]}))
        if a_out.shape != a_in.shape:
            out.append(result(f"{stem}: tabulated values (>=99.5 % of mapped cells, and of every class)", False, note="grid differs"))
            continue
        lut = np.full(256, -1, dtype=np.int16)
        for k, v in table.items():
            lut[k] = v
        expect = lut[a_in]
        m = expect >= 0
        got = np.asarray(a_out, dtype=float)[m]
        acc = float((got == expect[m]).mean()) if m.any() else 0.0
        # per input class: a whole class mapped wrongly or left unmapped must not hide inside
        # the pooled 0.5 % budget (open water 11 is 0.45 % of land-cover cells)
        a_m, g_all = a_in[m], np.asarray(a_out, dtype=float)[m]
        worst, bad = 1.0, []
        for k in np.unique(a_m):
            sel = a_m == k
            ck = float((g_all[sel] == lut[k]).mean())
            if ck < 0.995:
                bad.append(f"{int(k)}->{int(lut[k])}: {ck:.3f}")
            worst = min(worst, ck)
        ok = acc >= 0.995 and not bad
        out.append(result(f"{stem}: tabulated values (>=99.5 % of mapped cells, and of every class)", ok,
                          value={"pooled": round(acc, 5), "worst_class": round(worst, 5)}, threshold=0.995,
                          note=("classes wrong: " + ", ".join(bad)) if bad else ""))
    return out
