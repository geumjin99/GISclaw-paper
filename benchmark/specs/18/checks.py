"""Task 18 - percentage of street trees with NULL species, aggregated by quadtree region, SF.

Required:
  1. trees_count_vis.png exists and is not blank;
  3. (below) the regional null share is computed AND is what the map shows.
Reported, not required (needed only for route 3c):
  2. the species-nullity indicator was derived from the right column: a captured table with
     the 9 878 trees (+/-5 %) carries a 0/1, 0/100 (percent) or boolean column equal to Species.isnull() (or
     its complement notnull(), as in the geoplot gallery) on >= 99 % of rows (aligned on
     the input row index); nullity of another column (PlantDate 65 %, DBH 21 % ...) fails;
  3. the null share was aggregated by region and mapped, either
     (a) data-level: a captured polygon layer (4-5000 cells) has a column equal to the share
         of NULL (or non-NULL) species among the trees inside each cell, recomputed from the
         input Species column (fraction or %, |diff| <= 0.02 on >= 90 % of non-empty
         cells), AND that layer/column is the one drawn (drawn values, or face colours
         inverted to colormap positions and as monotone in the column as its classes allow,
         so classified schemes pass; cells drawn one by one with ax.add_patch count too);
         computing the share but mapping tree counts fails;
     (b) hexbin(C=nullity, reduce_C_function=mean): drawn hexagon values equal the share of
         NULL species among the trees assigned to each hexagon centre;
     (c) geoplot.quadtree(hue=<nullity column from 2>) with the default/mean aggregation
         (agg=np.sum gives counts of NULL trees, not a share, and fails). A user-defined agg
         (def or lambda) is evaluated on test 0/1 vectors and passes when it returns the mean
         or 100 x the mean.
     A tree-count quadtree, a map of the points, or a single city-wide percentage fail.
  4. NULL share (isnull) rather than its complement mapped. Required: the instruction says
     explicitly to map the percentage of trees whose species is NULL, so
     the geoplot-gallery notnull() map fails. For route 3c the sense is that of the hue column.
  5. (a method the task text prescribes is required) the map is a
     "quadtree format map": a geoplot.quadtree call, or the drawn polygon layer carrying the
     verified share (route 3a) has quadtree cells (rectangular cells in >= 2 sizes, each the
     largest cell's size / 2^k). Square-grid and hexbin aggregations fail this check.
"""
import os
import re

import numpy as np

from checklib import drawn_arrays, drawn_column_matches, file_exists, png_not_blank, result

NAME = "trees_count_vis.png"
N = 9878


def checks(ctx):
    import geopandas as gpd
    out = []
    r = file_exists(ctx, [NAME])
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(NAME))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))
    try:
        trees = gpd.read_file(os.path.join(ctx.inputs_dir, "street_trees_sample.geojson"))
        isnull = trees["Species"].isnull().to_numpy()
    except Exception as e:
        out.append(result("null share aggregated by region and mapped", False, note=repr(e)[:200]))
        return out

    # 2. per-tree indicator
    best, cols, sense = 0.0, set(), None
    col_sense = {}          # column -> "isnull" / "notnull" (>= 99 % agreement)
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("gdf", "df") or abs(meta.get("rows", 0) - N) > 0.05 * N:
            continue
        try:
            t = load()
        except Exception:
            continue
        idx = t["index"].to_numpy() if "index" in t.columns and t["index"].dtype.kind in "iu" else np.asarray(t.index)
        if not (np.issubdtype(np.asarray(idx).dtype, np.integer) and len(idx) and idx.min() >= 0 and idx.max() < N):
            continue
        ref = isnull[idx]
        for c in t.columns:
            if c in ("index", "geometry"):
                continue
            s = t[c]
            if s.dtype.kind not in "biuf":
                continue
            v = s.to_numpy().astype(float)
            vf = v[~np.isnan(v)]
            if np.isin(vf, [0, 1]).all():
                pass
            elif np.isin(vf, [0, 100]).all():
                v = v / 100.0          # percent indicator (isnull()*100)
            else:
                continue
            for sn, rr in (("isnull", ref), ("notnull", ~ref)):
                acc = float((v == rr).mean())
                if acc > best:
                    best, sense = acc, sn
                if acc >= 0.99:
                    cols.add(c)
                    col_sense.setdefault(c, sn)
    out.append(result("species nullity indicator from the Species column", best >= 0.99, value=round(best, 4),
                      threshold=0.99, required=False, note=f"columns {sorted(cols)} ({sense})" if cols else "no per-tree 0/1 column matching Species nullity"))

    # 3a. data-level regional aggregation on polygon layers -> verified (layer, column) pairs
    pts = trees[["geometry"]].copy()
    pts["isnull"] = isnull.astype(float)
    verified = {}           # (layer name, column) -> sense
    vlayers = {}            # layer name -> verified polygon layers (for the quadtree test)
    for tag, meta, load in ctx.objects("pred", "gdf"):
        if not (4 <= meta.get("rows", 0) <= 5000):
            continue
        try:
            g = load()
            if not g.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                continue
            g = g.reset_index(drop=True)
            if g.crs is None and meta.get("source") == "file":
                continue  # no CRS guessing for files written by the run
            p = pts.to_crs(g.crs) if g.crs is not None and pts.crs is not None and g.crs != pts.crs else pts
            j = gpd.sjoin(p, g[["geometry"]], predicate="within", how="inner")
            share = j.groupby("index_right")["isnull"].agg(["mean", "size"])
        except Exception:
            continue
        if len(share) < 4:
            continue
        for c in g.columns:
            if c == "geometry" or g[c].dtype.kind not in "fiu":
                continue
            v = g[c].astype(float).reindex(share.index).to_numpy()
            if not np.isfinite(v).any() or np.nanmax(np.abs(v)) > 100:
                continue
            for scale in (1.0, 100.0):
                for sn, ref in (("isnull", share["mean"].to_numpy()), ("notnull", 1 - share["mean"].to_numpy())):
                    ok = np.abs(v / scale - ref) <= 0.02
                    if float(np.nanmean(ok)) >= 0.9 and np.nanstd(ref) > 0:
                        verified[(meta["name"], c)] = sn
                        vlayers.setdefault(meta["name"], []).append(g)
    # ... and it must be the column that is DRAWN (values or colours of a figure collection)
    drawn = [(t, c) for t, c, _ in drawn_column_matches(ctx)]
    drawn += [(t, c) for t, c in _classed_color_matches(ctx)]
    agg_hits = [f"{t}.{c} ({verified[(t, c)]})" for t, c in drawn if (t, c) in verified]
    agg_sense = verified[[k for k in drawn if k in verified][0]] if agg_hits else None
    agg_note = "drawn: " + ", ".join(sorted(set(agg_hits))) if agg_hits else ""
    if verified and not agg_hits:
        agg_note = (f"null share computed ({sorted(verified)[:3]}) but the map draws another variable"
                    if drawn else f"null share computed ({sorted(verified)[:3]}) but no drawn choropleth column identified")
    # 3b. hexbin of the nullity (figure-only aggregation)
    hx_ok, hx_note, hx_sense = _hexbin_share(ctx, trees, isnull)
    if hx_ok:
        agg_note, agg_sense = hx_note, hx_sense
    # 3c. geoplot quadtree with hue = nullity column and a mean aggregation
    code = ctx.code_text() if hasattr(ctx, "code_text") else ""
    gp = False
    gp_sense = None
    for m in re.finditer(r"quadtree\s*\(", code):
        args = _split_args(code[m.end():])
        kw = {a.split("=", 1)[0].strip(): a.split("=", 1)[1].strip() for a in args if re.match(r"^\w+\s*=[^=]", a)}
        h = re.fullmatch(r"['\"]([^'\"]+)['\"]", kw.get("hue", ""))
        agg = kw.get("agg", "np.mean")
        if h and h.group(1) in cols and (re.search(r"mean|average", agg) or _agg_is_share(code, agg)):
            gp = True
            gp_sense = col_sense.get(h.group(1))
            agg_note = agg_note or f"geoplot.quadtree(hue={h.group(1)!r}, agg={agg})"
        elif h and h.group(1) in cols:
            agg_note = agg_note or f"geoplot.quadtree(hue={h.group(1)!r}) aggregates with agg={agg}, not a share"
    ok = bool(agg_hits) or hx_ok or gp
    out.append(result("null share aggregated by region and mapped", ok,
                      note=agg_note or "no drawn regional null share (layer column, hexbin or quadtree(hue=<nullity>))"))
    s = agg_sense or gp_sense or (sense if ok else None)
    out.append(result("NULL share (not its complement) mapped", s == "isnull", note=str(s)))
    # 5. the text prescribes "a quadtree format map" -> the regions must be
    # quadtree cells: geoplot.quadtree, or a drawn polygon layer carrying the verified share whose
    # rectangular cells have >= 2 sizes that are the largest cell's size / 2^k (a regular grid or
    # hexbin has one size and fails)
    q_calls = _geoplot_calls(code, ("quadtree",))
    q_ok, q_note = bool(q_calls), "geoplot.quadtree call" if q_calls else ""
    if not q_ok:
        for t, c in drawn:
            if (t, c) not in verified:
                continue
            for g in vlayers.get(t, []):
                good, info = _is_quadtree(g)
                q_note = f"{t}: {info}"
                if good:
                    q_ok = True
                    break
            if q_ok:
                break
    out.append(result("regions are quadtree cells (task: 'quadtree format map')", q_ok,
                      note=q_note or "no geoplot.quadtree call and no drawn share layer with quadtree cells "
                                     "(grid / hexbin / other regions)"))
    return out


_AGG_PROBE = r"""
import sys, json, numpy as np, pandas as pd
ns = {"np": np, "numpy": np, "pd": pd}
exec(sys.stdin.read(), ns)
f = ns["_agg"]
out = []
for v in ([0, 1, 1, 0, 1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 1, 0, 0, 1, 0, 0, 0]):
    r = []
    for x in (np.array(v, float), pd.Series(v, dtype=float)):
        try:
            r.append(float(f(x)))
        except Exception:
            r.append(None)
    out.append([float(np.mean(v)), r])
print(json.dumps(out))
"""


def _agg_is_share(code, agg):
    """A user-defined geoplot agg (a function defined in the code, or a lambda) that returns the
    mean of a 0/1 vector, as a fraction or in percent (e.g. 100*sum/len; the function need not
    be named mean/average). The function's own source
    (the def or lambda, nothing else of the run) is evaluated on three test vectors in an
    isolated subprocess (python -I, 10 s); every call must return mean or 100*mean."""
    import ast
    import json
    import subprocess
    import sys
    agg = agg.strip()
    src = None
    if agg.startswith("lambda"):
        src = "_agg = " + agg
    elif re.fullmatch(r"[A-Za-z_]\w*", agg):
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == agg:
                src = ast.unparse(n) + f"\n_agg = {agg}"
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda) \
                    and any(isinstance(t, ast.Name) and t.id == agg for t in n.targets):
                src = "_agg = " + ast.unparse(n.value)
    if not src:
        return False
    try:
        r = subprocess.run([sys.executable, "-I", "-c", _AGG_PROBE], input=src, capture_output=True,
                           text=True, timeout=10)
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return False
    for m, vals in res:
        for v in vals:
            if v is None or not any(abs(v - k * m) <= 1e-9 for k in (1.0, 100.0)):
                return False
    return True


def _is_quadtree(g):
    """(ok, info): the layer's (near-)rectangular cells come in >= 2 sizes, each the largest
    cell's width and height divided by the same power of two (tolerance 0.05 in log2, >= 90 % of
    rectangular cells; clipped edge cells are not rectangular and are ignored)."""
    try:
        b = g.geometry.bounds
        w, h = (b.maxx - b.minx).to_numpy(float), (b.maxy - b.miny).to_numpy(float)
        a = g.geometry.area.to_numpy(float)
        rect = (w > 0) & (h > 0) & (a >= 0.98 * w * h)
        if rect.sum() < 4:
            return False, f"{int(rect.sum())} rectangular cells"
        w, h = w[rect], h[rect]
        kw, kh = np.log2(w.max() / w), np.log2(h.max() / h)
        on = (np.abs(kw - np.round(kw)) <= 0.05) & (np.abs(kh - np.round(kh)) <= 0.05) & (np.round(kw) == np.round(kh))
        levels = sorted(set(np.round(kw[on]).astype(int).tolist()))
        info = {"rect_cells": int(rect.sum()), "on_power2_share": round(float(on.mean()), 3), "levels": levels[:8]}
        return bool(on.mean() >= 0.9 and len(levels) >= 2), info
    except Exception as e:
        return False, repr(e)[:80]


def _geoplot_calls(code, funcs):
    """Names of geoplot functions in `funcs` called in the code (comments removed); module aliases
    from `import geoplot [as x]` / `import geoplot.geoplot as x`, `gplt` / `geoplot` always
    accepted (pre-imported in some settings), `from geoplot import quadtree [as f]` followed."""
    code = "\n".join("" if re.match(r"\s*#", ln) else ln.split("#")[0] for ln in code.splitlines())
    mods, names = {"gplt", "geoplot"}, set()
    for m in re.finditer(r"^\s*import\s+(.+)$", code, re.M):
        for part in m.group(1).split(","):
            mm = re.match(r"\s*(geoplot(?:\.geoplot)?)\s*(?:as\s+(\w+))?\s*$", part)
            if mm:
                mods.add(mm.group(2) or mm.group(1))
    for m in re.finditer(r"^\s*from\s+geoplot(?:\.geoplot)?\s+import\s+\(?([^)\n]+)", code, re.M):
        for part in m.group(1).split(","):
            mm = re.match(r"\s*(\w+)\s*(?:as\s+(\w+))?", part)
            if mm and mm.group(1) in funcs:
                names.add(mm.group(2) or mm.group(1))
    fn = "|".join(funcs)
    hits = set()
    for mod in mods:
        hits.update(re.findall(r"(?<![\w.])" + re.escape(mod) + r"\.(" + fn + r")\s*\(", code))
    for n in names:
        if re.search(r"(?<![\w.])" + re.escape(n) + r"\s*\(", code):
            hits.add(n)
    return sorted(hits)


def _split_args(s):
    depth, cur, out, quote = 0, "", [], None
    for ch in s:
        if quote:
            cur += ch
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                break
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def _classed_color_matches(ctx, min_ratio=0.98):
    """(table, column) drawn as a choropleth (GeoDataFrame.plot passes colours): face colours
    are inverted to colormap positions; a column matches when its values are as monotone
    in those positions as possible given the classes (continuous or classified schemes)."""
    from scipy.stats import spearmanr
    from checklib import _cmap_positions
    inv = []
    for t, m, fc in drawn_arrays(ctx, "pred"):
        if not m["name"].endswith("_facecolors"):
            continue
        fc = np.asarray(fc, float)
        if fc.ndim != 2 or len(np.unique(np.round(fc[:, :3], 4), axis=0)) < 3:
            continue
        try:
            pos, _, res = _cmap_positions(fc)
        except Exception:
            continue
        if pos is not None and res <= 0.02:
            inv.append(np.asarray(pos, float))
    # cells drawn one by one with ax.add_patch (no collection): face colours of each hand-drawn
    # patch group (capture.drawn_patch_groups), inverted the same way
    from capture import drawn_patch_groups
    root = getattr(ctx, "pred_snap", None)
    if root and os.path.isdir(root):
        for t in sorted(os.listdir(root)):
            if not t.startswith("save"):
                continue
            for g in drawn_patch_groups(os.path.join(root, t)):
                fc = np.asarray(g["facecolors"], float)
                if not g.get("fill") or fc.ndim != 2 or len(fc) < 3 or len(np.unique(np.round(fc[:, :3], 4), axis=0)) < 3:
                    continue
                try:
                    pos, _, res = _cmap_positions(fc)
                except Exception:
                    continue
                if pos is not None and res <= 0.02:
                    inv.append(np.asarray(pos, float))
    hits = []
    if not inv:
        return hits
    for tag, meta, load in ctx.objects("pred", "gdf", include_files=False):
        try:
            g = load()
            parts = np.array([len(getattr(x, "geoms", [x])) if x is not None else 1 for x in g.geometry])
        except Exception:
            continue
        for c in g.columns:
            if c == "geometry" or g[c].dtype.kind not in "fiu":
                continue
            raw = g[c].astype(float).to_numpy()
            for v in (raw[np.isfinite(raw)], np.repeat(raw, parts)[np.isfinite(np.repeat(raw, parts))]):
                hit = False
                for pos in inv:
                    if len(v) != len(pos) or np.std(v) == 0 or np.std(pos) == 0:
                        continue
                    rho = spearmanr(v, pos).correlation
                    # best rho achievable with these class sizes: classes filled in value order
                    _, inv_idx, cnt = np.unique(np.round(pos, 4), return_inverse=True, return_counts=True)
                    ideal = np.repeat(np.arange(len(cnt)), cnt)
                    rmax = spearmanr(np.sort(v), ideal).correlation
                    if rho >= min_ratio * rmax and rho > 0.8:
                        hit = True
                        break
                if hit:
                    hits.append((meta["name"], c))
                    break
    return hits


def _hexbin_share(ctx, trees, isnull):
    """hexbin(C=nullity, reduce=mean): drawn PolyCollection values vs the share of NULL
    species among the trees nearest to each hexagon centre (matplotlib's hex assignment)."""
    from scipy.spatial import cKDTree
    crss = {"EPSG:4326"}
    for tag, meta, load in ctx.objects("pred", "gdf", include_files=False):
        if meta.get("crs") and meta["crs"] != "None":
            crss.add(meta["crs"])
    vals = {(t, m["name"].rsplit("_", 1)[0]): np.asarray(a, float).ravel() for t, m, a in drawn_arrays(ctx, "pred")
            if m.get("artist") == "PolyCollection" and m["name"].endswith("_values")}
    for t, m, off in drawn_arrays(ctx, "pred"):
        key = (t, m["name"].rsplit("_", 1)[0])
        if m.get("artist") != "PolyCollection" or not m["name"].endswith("_offsets") or key not in vals:
            continue
        off = np.asarray(off, float)
        v = vals[key]
        if len(v) != len(off) or off.ndim != 2 or off.shape[1] != 2:
            continue
        fin = np.isfinite(off).all(axis=1)   # NaN offsets would make the lattice NaN
        off, v = off[fin], v[fin]
        if len(v) < 4:
            continue
        ux, uy = np.diff(np.unique(np.round(off[:, 0], 6))), np.diff(np.unique(np.round(off[:, 1], 6)))
        if not len(ux) or not len(uy):
            continue
        sx, sy = 2 * ux[ux > 0].min(), 2 * uy[uy > 0].min()
        for crs in crss:
            try:
                p = trees.geometry.to_crs(crs)
            except Exception:
                continue
            xy = np.c_[p.x, p.y]
            fin_t = np.isfinite(xy).all(axis=1)
            if not fin_t.any():
                continue
            xy, isn = xy[fin_t], np.asarray(isnull)[fin_t]
            inb = (xy[:, 0] >= off[:, 0].min() - sx) & (xy[:, 0] <= off[:, 0].max() + sx) & \
                  (xy[:, 1] >= off[:, 1].min() - sy) & (xy[:, 1] <= off[:, 1].max() + sy)
            if inb.mean() < 0.9:
                continue
            # matplotlib hexbin: two lattices (x0 + i sx, y0 + j sy) and the same shifted by
            # (sx/2, sy/2); a point goes to the nearest centre in (x/sx, sqrt(3) y/sy) space.
            # Rebuild the full lattice (cells hidden by mincnt still capture their trees).
            bx, by = off[0]
            i0 = np.floor((xy[:, 0].min() - bx) / sx) - 2
            i1 = np.ceil((xy[:, 0].max() - bx) / sx) + 2
            j0 = np.floor((xy[:, 1].min() - by) / sy) - 2
            j1 = np.ceil((xy[:, 1].max() - by) / sy) + 2
            if (i1 - i0) * (j1 - j0) > 4e6:
                continue
            I, J = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
            lat = np.r_[np.c_[bx + I.ravel() * sx, by + J.ravel() * sy],
                        np.c_[bx + (I.ravel() + 0.5) * sx, by + (J.ravel() + 0.5) * sy]]
            scale_xy = np.array([1 / sx, np.sqrt(3) / sy])
            _, li = cKDTree(lat * scale_xy).query(xy * scale_xy)
            n_l = np.bincount(li, minlength=len(lat))
            k_l = np.bincount(li, weights=isn.astype(float), minlength=len(lat))
            _, oi = cKDTree(lat * scale_xy).query(off * scale_xy)
            n, k = n_l[oi], k_l[oi]
            share = np.where(n > 0, k / np.maximum(n, 1), np.nan)
            m_ = np.isfinite(v) & (n > 0)
            for scale in (1.0, 100.0):
                for sn, ref in (("isnull", share), ("notnull", 1 - share)):
                    frac = float((np.abs(v[m_] / scale - ref[m_]) <= 0.02).mean()) if m_.any() else 0.0
                    if frac >= 0.9 and np.nanstd(ref[m_]) > 0:
                        return True, f"hexbin {key[1]}@{t} ({m_.sum()} hexagons, {sn}, x{scale:g})", sn
    return False, "", None
