"""Task 36 - hail damage: mean SAVI loss per crop field.

Gold: SAVI (L = 0.5, reflectance) before minus after on the image overlap, negative loss
remapped to NoData, mean per field; fields are Web-Mercator coordinates in a shapefile with
no .shx/.dbf/.prj (row order = field id); 287 of 319 fields fall in the overlap.

Correctness (required):
  1. some result file written to pred_results (the task names none);
  2. a per-field loss result exists: a captured field table, a dict/Series keyed by field id,
     or a CSV/vector file written to pred_results (aligned to the input fields by geometry -
     UTM 12N or Web Mercator - an id column, a dict key or a full row index) with values for
     >= 80 % of the gold fields. Numeric field ids may be 0-based (row order) or 1-based
     (Field_ID = row + 1): both alignments are tried, a wrong one cannot correlate;
  3. its ranking agrees with gold: Spearman >= 0.85. Without the remap 0.99, DN-based SAVI or
     NDVI 0.93 pass this check; after-minus-before (sign) -0.28, wrong bands (B1/B2) 0.24 fail; fields
     left in the wrong CRS get no values at all.
  4. (the task text names the Soil-Adjusted Vegetation Index) SAVI
     with L = 0.5 used (code inspection for the soil factor); NDVI loss ranks close to gold
     (0.93) and is rejected only here.
"""
import os
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

from checklib import result, key_values


def _fields():
    tmp = tempfile.mkdtemp()
    import geopandas as gpd
    try:
        shutil.copy(os.path.join(_fields.inputs, "Field_boundaries.shp"), os.path.join(tmp, "f.shp"))
        os.chmod(os.path.join(tmp, "f.shp"), 0o644)
        old = os.environ.get("SHAPE_RESTORE_SHX")
        os.environ["SHAPE_RESTORE_SHX"] = "YES"
        f = gpd.read_file(os.path.join(tmp, "f.shp"))
        if old is None:
            os.environ.pop("SHAPE_RESTORE_SHX", None)
        f = f.set_crs(3857, allow_override=True)
        f["_fid"] = np.arange(len(f))
        return f
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def checks(ctx):
    out = []
    files = ctx.nonempty_pred_files()   # 0-byte files are not results
    out.append(result("result file written to pred_results", bool(files), value=files[:6], threshold="any non-empty file"))
    nm = ["per-field loss result covers >= 80 % of gold fields", "per-field loss ranking matches gold (Spearman >= 0.85)"]
    gold = None
    for tag, meta, load in ctx.objects("gold", "gdf"):
        if meta["name"] == "field_loss":
            g = load()
            gold = pd.Series(g["MEAN_veg_loss"].values, index=g["Field_ID"].values - 1).dropna()
            break
    try:
        _fields.inputs = ctx.inputs_dir
        fields = _fields()
    except Exception as e:
        fields = None
    if gold is None or fields is None:
        for n in nm:
            out.append(result(n, False, note="gold or field layer unavailable"))
        return out
    fields_alt = fields.to_crs(32612)
    best = None

    def _score(s, lab):
        nonlocal best
        common = gold.index.intersection(s.index)
        cov = len(common) / len(gold)
        if len(common) < 10 or s.loc[common].std() == 0:
            return
        rho = float(s.loc[common].rank().corr(gold.loc[common].rank()))
        sc = (cov >= 0.8 and rho >= 0.85, rho)
        if best is None or sc > best[0]:
            best = (sc, lab, cov, rho)
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("gdf", "df", "series") or not (250 <= meta.get("rows", 0) <= 340):
            continue
        try:
            obj = load()
        except Exception:
            continue
        if meta.get("kind") == "series":
            # dict / Series: 'index' column holds the field ids, the other column the values
            try:
                idx = pd.to_numeric(obj["index"], errors="coerce")
                if idx.isna().mean() > 0.2 or (idx.dropna() % 1 != 0).any():
                    continue
                vcol = [c for c in obj.columns if c != "index"][0]
                for off in (0, 1):
                    s = pd.Series(obj[vcol].astype(float).to_numpy(), index=(idx - off).to_numpy()).dropna()
                    s = s[np.isfinite(s)].groupby(level=0).mean()
                    _score(s, f"{meta['name']}@{tag} (dict ids{'-1' if off else ''})")
            except Exception:
                pass
            continue
        if meta.get("source") == "file" and "geometry" in obj.columns and getattr(obj, "crs", None) is None:
            continue  # no CRS guessing for files written by the run
        keys = None
        for ref in (fields_alt, fields):
            try:
                o = obj
                if "geometry" in obj.columns and getattr(obj, "crs", None) is None:
                    import geopandas as gpd
                    o = gpd.GeoDataFrame(obj.copy(), geometry="geometry", crs=ref.crs)  # undeclared CRS (in-memory only): try both
                keys = key_values(o, ref, "_fid")
            except Exception:
                keys = None
            if keys is not None:
                break
        if keys is None and len(obj) == len(fields):
            keys = pd.Series(np.arange(len(obj)), index=obj.index)
        if keys is None:
            continue
        geom_keys = "geometry" in obj.columns
        offsets = (0,) if geom_keys else (0, 1)   # numeric ids may be 1-based Field_IDs
        for c in obj.columns:
            if str(c).lower() in ("geometry", "_fid", "index", "level_0", "field_id", "fid", "id", "count", "n_pixels",
                                  "area", "shape_area", "pixels", "n") or obj[c].dtype.kind not in "fi":
                continue
            for off in offsets:
                k = pd.to_numeric(keys, errors="coerce") - off
                s = obj[c].astype(float).groupby(k.values).mean()
                _score(s[np.isfinite(s)], f"{meta['name']}.{c}@{tag}" + (" (ids-1)" if off else ""))
    if best is None:
        out.append(result(nm[0], False, note="no field-level table / dict / file (250-340 rows) with values"))
        out.append(result(nm[1], False))
    else:
        _, lab, cov, rho = best
        out.append(result(nm[0], cov >= 0.8, value=round(cov, 3), threshold=0.8, note=lab))
        out.append(result(nm[1], rho >= 0.85, value=round(rho, 3), threshold=0.85, note=lab))
    code = ctx.code_text()
    savi = bool(re.search(r"\+\s*0\.5\s*\)|L\s*=\s*0\.5|\*\s*1\.5|1\s*\+\s*L", code))
    # The task text names SAVI as the index to use -> required.
    out.append(result("SAVI (L = 0.5) used", savi, note="code inspection (soil factor L = 0.5 / (1 + L) / * 1.5)"))
    return out
