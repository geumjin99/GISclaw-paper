"""T05 - Elevated blood lead levels in children (Sacramento): privacy-preserving hot spots.

Task core: Optimized Hot Spot Analysis (Getis-Ord Gi* on incident counts aggregated to
hexagons), a hexagonal tessellation summarising the number of cases per hexagon, and a map
showing only the aggregated hexagons. The point file has no attributes: each point = 1 case.

All checks are computed on the solution's own hexagon layer (any cell size, CRS or orientation),
so legitimate choices of scale are accepted:
Correctness (required):
  1. the map (instructed name 'blood_lead_hotspots.png'; 'transit_access.png', the name in the
     published instruction, and any blood-lead-specific .png are also accepted)
     exists and is not blank;
  2. hexagonal tessellation with correct case counts: a captured polygon layer, >= 50 % of whose
     cells (or of whose full-size cells, for a grid clipped to the study area) are
     hexagons (6 vertices), carries a column equal to the number of case points inside
     each cell (re-counted here with a spatial join) for >= 95 % of cells, and the cells hold
     >= 90 % of the 373 cases;
  3. hot spot statistic: on a hexagon layer, a per-cell column (or a captured array of the same
     length) that is not just the count (Spearman with counts < 0.97) rank-correlates >= 0.6 with
     a Getis-Ord Gi* z-score recomputed here on that layer (distance band ~2 hexagon spacings)
     AND is a Gi*-type product: signed confidence bins (integers in -3..3, >= 3 levels or both
     signs; a 0/1 flag is not) or a signed z-scaled
     statistic (both signs, spread within 5x of the recomputed z). Gi* with other bands / kNN
     and hot-cold bins pass; raw counts, kernel densities, local Moran's I fail;
  4. the map draws that hot spot result: a drawn hexagon collection whose colour values equal a
     passing column (or whose explicit facecolours, <= 12 classes, explain it: eta^2 >= 0.9), or
     whose colour classes (<= 12, values or explicit facecolours) are ordered intervals of it,
     e.g. FDR confidence bins of the Gi* z drawn one colour per bin; when
     the layer carries a p-value column the class order may follow sign(z)*(1-p) instead of z
     (permutation-p classes).
     A collection drawing only part of the layer (e.g. all cells grey, significant ones coloured
     on top) also passes when its patches map back by geometry to exactly the significant cells
     (p <= 0.01/0.05/0.10, |z| >= 1.645/1.96/2.576, or non-zero bins; optionally hot-only or
     cold-only) and its colour values equal the statistic there - see _drawn_subset;
     a map drawn one class at a time (one single-colour collection per class) is judged on all
     collections of the axes merged - see _drawn_merged;
  5. privacy: fewer than 20 drawn point markers sit on case locations (figure offsets tested
     against the cases in every CRS used in the run).
Reported, not required:
  6. method fidelity: a Getis-Ord function is called (not merely imported);
  7. privacy: median cell area >= 0.05 km2 (cells large enough not to reveal homes).
"""
import numpy as np

from checklib import file_exists, png_not_blank, result, drawn_arrays

N_CASES = 373


def _hexness(g):
    """Share of six-vertex cells. A hexagon grid clipped to
    the study area has edge cells that are no longer hexagons; the share is therefore also taken
    over the full-size cells only (area >= 0.95 x the 90th-percentile cell area, at least 10 cells
    and 25 % of the layer) and the larger of the two shares is returned. A square fishnet or
    other polygons have no six-vertex full cells either way."""
    def nv(geom):
        if geom is None or geom.geom_type != "Polygon":
            return 0
        c = np.asarray(geom.simplify(1e-6 * max(1.0, np.sqrt(geom.area))).exterior.coords)[:-1]
        return len(c)
    six = np.array([nv(x) == 6 for x in g.geometry], bool)
    share = float(np.mean(six)) if len(six) else 0.0
    try:
        a = np.asarray(g.geometry.area, float)
        full = np.isfinite(a) & (a >= 0.95 * np.nanpercentile(a, 90))
        if full.sum() >= max(10, 0.25 * len(a)):
            share = max(share, float(np.mean(six[full])))
    except Exception:
        pass
    return share


def _gistar_z(g, counts):
    import libpysal
    from esda.getisord import G_Local
    from scipy.spatial import cKDTree
    cent = np.c_[g.geometry.centroid.x, g.geometry.centroid.y]
    # missing / empty cells have no centroid (NaN): z only for the cells that have one
    ok = np.isfinite(cent).all(axis=1)
    z = np.full(len(cent), np.nan)
    if ok.sum() < 3:
        return z
    cent = cent[ok]
    d = cKDTree(cent).query(cent, k=2)[0][:, 1]
    band = 2.05 * np.median(d)
    w = libpysal.weights.DistanceBand(cent, threshold=band, binary=True, silence_warnings=True)
    z[ok] = G_Local(np.asarray(counts, float)[ok], w, star=True, permutations=0).Zs
    return z


def _spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or a[m].std() == 0 or b[m].std() == 0:
        return float("nan")
    return float(spearmanr(a[m], b[m]).correlation)


def _zlike(v, z):
    """Gi*-type product: signed confidence bins (integers within -3..3; >= 3 levels or both
    signs) or a signed, z-scaled statistic (both signs, spread within 5x of the recomputed z).
    Densities/KDE values (all positive, arbitrary scale) and 0/1 flags are not."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return False
    if np.allclose(v, np.round(v)) and v.min() >= -3 and v.max() <= 3:
        # a two-level 0/1 flag (e.g. "overlaps a hot
        # area") is not a set of confidence bins: bins need >= 3 levels or both signs
        return bool(len(np.unique(v)) >= 3 or ((v < 0).any() and (v > 0).any()))
    zs = float(np.nanstd(z)) or 1.0
    return bool((v < 0).any() and (v > 0).any() and 0.2 <= float(v.std()) / zs <= 5)


def _nparts(geom):
    try:
        return len(geom.geoms) if hasattr(geom, "geoms") else 1
    except Exception:
        return 1


def _monotone_classes(stat, lab, max_viol=0.02):
    """Are the drawn classes ordered intervals of `stat` (a binning of it)? Classes are ranked
    by their median statistic; with cells sorted by the statistic, the class rank may decrease
    for at most 2 % of the cells (longest non-decreasing subsequence). A map of counts or
    densities is not a binning of the Gi* statistic and fails."""
    import bisect
    ok = np.isfinite(stat)
    x, l = stat[ok], np.asarray(lab)[ok]
    ks = np.unique(l)
    if len(ks) < 2 or len(x) < 10:
        return False
    med = {k: float(np.median(x[l == k])) for k in ks}
    order_k = sorted(ks, key=lambda k: med[k])
    rank = np.array([order_k.index(k) for k in l])
    seq = rank[np.lexsort((rank, x))]       # by statistic; ties ordered by class
    tails = []
    for r in seq:
        i = bisect.bisect_right(tails, r)
        if i == len(tails):
            tails.append(r)
        else:
            tails[i] = r
    return (len(seq) - len(tails)) <= max_viol * len(seq)


def _drawn_stat(ctx, layer, stat):
    """Is `stat` (per-cell values of `layer`) what a drawn polygon collection shows?
    (a) drawn colour values equal the statistic (in patch order; multi-part cells repeat), or
    (b) explicit facecolours (color=...): <= 12 distinct colours that explain the statistic
    (correlation ratio eta^2 >= 0.9), or
    (c) the drawn colours - colour values or <= 12 explicit facecolours -
    are a classification of the statistic: ordered intervals of it, e.g. FDR confidence bins of
    a Gi* z-score drawn with one colour per bin (see _monotone_classes).
    If the layer carries a p-value column, the classes may instead be ordered in
    sign(stat) * (1 - p): significance classes from permutation p-values are not perfectly
    monotone in the analytic z."""
    parts = np.array([_nparts(g) for g in layer.geometry])
    stat = np.asarray(stat, float)
    ok = np.isfinite(stat)
    orders = [stat] + [np.sign(stat) * (1 - pv) for pv in _p_columns(layer)]
    starts = np.r_[0, np.cumsum(parts)[:-1]]
    drawn = drawn_arrays(ctx, "pred")
    for tag, meta, a in drawn:
        if meta.get("artist") != "PatchCollection":
            continue
        a = np.asarray(a, float)
        if meta["name"].endswith("_values"):
            d = a.ravel()
            for sel in (np.ones(len(stat), bool), ok):
                exp = np.repeat(stat[sel], parts[sel])
                if len(exp) == len(d) and np.allclose(exp, d, rtol=1e-6, atol=1e-9, equal_nan=True):
                    return meta["name"] + "@" + tag
            if len(d) == parts.sum():
                cell = d[starts]
                m = np.isfinite(cell) & ok
                if 2 <= len(np.unique(cell[m])) <= 12 and any(_monotone_classes(o[m], cell[m]) for o in orders):
                    return meta["name"] + "@" + tag + " (classes of the statistic)"
        elif meta["name"].endswith("_facecolors") and a.ndim == 2 and len(a) == parts.sum():
            cols = [tuple(np.round(c, 3)) for c in a[starts]]
            uniq = sorted(set(cols))
            if not 2 <= len(uniq) <= 12:
                continue
            lab = np.array([uniq.index(c) for c in cols])
            x, l = stat[ok], lab[ok]
            tot = ((x - x.mean()) ** 2).sum()
            if tot == 0:
                continue
            within = sum(((x[l == k] - x[l == k].mean()) ** 2).sum() for k in np.unique(l))
            if 1 - within / tot >= 0.9:
                return meta["name"] + "@" + tag + " (facecolours)"
            if any(_monotone_classes(o, lab) for o in orders):
                return meta["name"] + "@" + tag + " (facecolour classes of the statistic)"
    return _drawn_subset(layer, stat, drawn) or _drawn_merged(layer, stat, drawn, orders)


def _eta2(x, lab):
    tot = ((x - x.mean()) ** 2).sum()
    if tot == 0:
        return 0.0
    within = sum(((x[lab == k] - x[lab == k].mean()) ** 2).sum() for k in np.unique(lab))
    return 1 - within / tot


def _class_sig_iou(layer, stat, lab):
    """Best IoU between the drawn hot / cold classes
    and a significant cell set of _significant_masks. Classes are ranked by their median
    statistic; a hot-only set is compared with the upper tails of the ranking, a cold-only set
    with the lower tails, a two-sided set with the classes outside a middle band (the band holding
    the class of smallest median |statistic|, i.e. 'not significant'). A 2-class split whose small
    class is under 2 % of the cells is trivially 'monotone'; this asks that the drawn extreme
    classes actually show the significant cells (a map that outlines 2 of 269 significant hot
    hexagons over a grey 'suppressed' layer does not)."""
    masks = _significant_masks(layer, stat)
    if not masks:
        return 0.0
    ks = np.unique(lab)
    med = {k: float(np.nanmedian(stat[lab == k])) if np.isfinite(stat[lab == k]).any() else 0.0 for k in ks}
    order = sorted(ks, key=lambda k: med[k])
    n = len(order)
    ns = min(range(n), key=lambda i: abs(med[order[i]]))
    upper = [np.isin(lab, order[i:]) for i in range(1, n)]
    lower = [np.isin(lab, order[:i]) for i in range(1, n)]
    outside = [np.isin(lab, order[:i] + order[j:]) for i in range(0, ns + 1) for j in range(ns + 1, n + 1)
               if i < j and (i > 0 or j < n)]
    best = 0.0
    for m in masks:
        sm = stat[m]
        sets = upper if (sm > 0).all() else lower if (sm < 0).all() else outside
        for u in sets:
            un = (u | m).sum()
            if un:
                best = max(best, float((u & m).sum() / un))
    return best


def _drawn_merged(layer, stat, drawn, orders):
    """A class map drawn one
    class at a time (for c in classes: layer[layer.cls == c].plot(color=...)) gives one
    single-colour PatchCollection per class, which the tests above see one at a time. Here all
    PatchCollections of one axes whose every patch maps back to a layer cell (bounding-box centre
    within 5 % of the cell size, as in _drawn_subset) are merged into one drawing; a cell drawn
    twice keeps the colour of the later collection (drawn on top). A cell's colour class is its
    explicit facecolour or drawn colour value where captured, otherwise the identity of its
    single-colour collection (the capture keeps no facecolour for a one-colour collection). The
    merged drawing is then judged exactly as one collection: all cells drawn, 2-12 classes that
    explain the statistic (eta^2 >= 0.9) or are ordered intervals of it (_monotone_classes) and
    a set of classes that matches a significant cell set, IoU >= 0.5
    (_class_sig_iou); or
    only the cells of a significant set (_significant_masks) drawn, >= 3 of them, 2-12 classes
    with eta^2 >= 0.9 (a one-colour overlay of the significant cells is accepted only together
    with a drawn background of the other cells, i.e. via the all-cells branch)."""
    import re
    from scipy.spatial import cKDTree
    try:
        pc, owner = [], []
        for i, geom in enumerate(layer.geometry):
            for part in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                b = part.bounds
                pc.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                owner.append(i)
        pc, owner = np.asarray(pc, float), np.asarray(owner)
        size = float(np.sqrt(np.nanmedian(layer.geometry.area)))
        tree = cKDTree(pc)
    except Exception:
        return None
    if not np.isfinite(size) or size <= 0:
        return None
    stat = np.asarray(stat, float)
    by = {}
    for tag, meta, a in drawn:
        m = re.match(r"fig_ax(\d+)_coll(\d+)_(paths_bbox|facecolors|values)$", meta.get("name", ""))
        if m and meta.get("artist") == "PatchCollection":
            by.setdefault((tag, int(m.group(1))), {}).setdefault(int(m.group(2)), {})[m.group(3)] = np.asarray(a, float)
    for (tag, ax), colls in by.items():
        lab = {}
        used = 0
        for j in sorted(colls):
            arrs = colls[j]
            bb = arrs.get("paths_bbox")
            if bb is None or bb.ndim != 2 or bb.shape[1] != 4 or not len(bb):
                continue
            cen = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
            if not np.isfinite(cen).all():
                continue
            d, k = tree.query(cen)
            if (d > 0.05 * size).any():
                continue                         # not (only) cells of this layer: another layer
            fc, v = arrs.get("facecolors"), arrs.get("values")
            if v is not None and v.ravel().shape[0] == len(bb):
                keys = [("v", round(float(x), 9)) for x in v.ravel()]
            elif fc is not None and fc.ndim == 2 and len(fc) == len(bb):
                keys = [("c",) + tuple(np.round(c, 3)) for c in fc]
            else:
                keys = [("coll", j)] * len(bb)
            per = {}
            for cell, key in zip(owner[k], keys):
                per.setdefault(int(cell), set()).add(key)
            if any(len(x) > 1 for x in per.values()):
                continue                         # parts of one cell in different colours
            for cell, x in per.items():
                lab[cell] = next(iter(x))        # later collection drawn on top
            used += 1
        if used < 2 or not lab:
            continue
        cells = np.array(sorted(lab))
        keys = sorted(set(lab.values()), key=str)
        l = np.array([keys.index(lab[c]) for c in cells])
        if not 2 <= len(keys) <= 12:
            continue
        if len(cells) == len(stat):
            ok = np.isfinite(stat)
            if ok.sum() >= 10 and (_eta2(stat[ok], l[ok]) >= 0.9 or any(_monotone_classes(o, l) for o in orders)):
                iou = _class_sig_iou(layer, stat, l)
                if iou >= 0.5:
                    return f"{used} collections of ax{ax}@{tag} merged (one per colour class; significant-set IoU {iou:.2f})"
        else:
            sel = np.zeros(len(stat), bool)
            sel[cells] = True
            if len(cells) >= 3 and any((sel == m).all() for m in _significant_masks(layer, stat)) \
                    and _eta2(stat[cells], l) >= 0.9:
                return f"{used} collections of ax{ax}@{tag} merged (significant cells only, {len(cells)} of {len(stat)})"
    return None


def _significant_masks(layer, stat):
    """Cell sets a hot spot map may legitimately restrict itself to: the significant cells
    (p <= 0.01/0.05/0.10 on a p-value column of the layer; |z| >= 1.645/1.96/2.576 for a z-scaled
    statistic; non-zero confidence bins), optionally only the hot (> 0) or only the cold (< 0) ones."""
    stat = np.asarray(stat, float)
    base = []
    for pv in _p_columns(layer):
        base += [pv <= a for a in (0.01, 0.05, 0.10)]
    f = stat[np.isfinite(stat)]
    if len(f) and np.allclose(f, np.round(f)) and f.min() >= -3 and f.max() <= 3:
        base.append(stat != 0)
    else:
        base += [np.abs(stat) >= t for t in (1.645, 1.96, 2.576)]
    out = []
    for m in base:
        m = np.asarray(m, bool) & np.isfinite(stat)
        out += [m, m & (stat > 0), m & (stat < 0)]
    return [m for m in out if m.any()]


def _drawn_subset(layer, stat, drawn):
    """A map may draw every cell in grey and colour only
    the significant ones. A drawn PatchCollection with FEWER patches than the layer is matched
    back to layer cells by geometry (bounding-box centre of each drawn patch = that of a layer
    cell part, within 5 % of the cell size); the matched cells must be exactly one of the
    significant cell sets of _significant_masks (an arbitrary or non-significant subset never
    passes), and the drawn colour values must equal the statistic on them (or <= 12 explicit
    facecolours explain it, eta^2 >= 0.9)."""
    from scipy.spatial import cKDTree
    try:
        pc, owner = [], []
        for i, geom in enumerate(layer.geometry):
            for part in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                b = part.bounds
                pc.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                owner.append(i)
        pc, owner = np.asarray(pc, float), np.asarray(owner)
        size = float(np.sqrt(np.median(layer.geometry.area)))
        tree = cKDTree(pc)
    except Exception:
        return None
    masks = _significant_masks(layer, stat)
    if not masks or not np.isfinite(size) or size <= 0:
        return None
    by = {}
    for tag, meta, a in drawn:
        if meta.get("artist") == "PatchCollection":
            by[(tag, meta["name"])] = np.asarray(a, float)
    for (tag, name), bb in by.items():
        if not name.endswith("_paths_bbox") or bb.ndim != 2 or bb.shape[1] != 4:
            continue
        if not 1 <= len(bb) < len(pc):
            continue
        cen = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
        if not np.isfinite(cen).all():
            continue
        d, j = tree.query(cen)
        if (d > 0.05 * size).any():
            continue
        cells = owner[j]
        uniq = np.unique(cells)
        sel = np.zeros(len(stat), bool)
        sel[uniq] = True
        if not any((sel == m).all() for m in masks):
            continue
        pre = name[: -len("_paths_bbox")]
        first = np.r_[True, cells[1:] != cells[:-1]]   # one entry per drawn cell
        if len(np.unique(cells[first])) != len(uniq):
            continue                                   # a cell's parts are not contiguous
        cell_order = cells[first]
        v = by.get((tag, pre + "_values"))
        if v is not None and v.ravel().shape[0] == len(cells):
            dv = v.ravel()[first]
            if np.allclose(dv, stat[cell_order], rtol=1e-6, atol=1e-9):
                return pre + "_values@" + tag + f" (significant cells only, {len(uniq)} of {len(stat)})"
        fc = by.get((tag, pre + "_facecolors"))
        if fc is not None and fc.ndim == 2 and len(fc) == len(cells) and len(uniq) >= 3:
            cols = [tuple(np.round(c, 3)) for c in fc[first]]
            u = sorted(set(cols))
            if 2 <= len(u) <= 12:
                lab = np.array([u.index(c) for c in cols])
                x = stat[cell_order]
                tot = ((x - x.mean()) ** 2).sum()
                if tot > 0:
                    within = sum(((x[lab == k] - x[lab == k].mean()) ** 2).sum() for k in np.unique(lab))
                    if 1 - within / tot >= 0.9:
                        return pre + "_facecolors@" + tag + f" (significant cells only, {len(uniq)} of {len(stat)})"
    return None


def _p_columns(layer):
    """Numeric p-value columns of a layer (name p / p_sim / gi_p / p_value / pval ...; values in
    [0, 1], not constant)."""
    import re
    out = []
    for c in layer.columns:
        if c == "geometry" or not re.search(r"(^|_)p($|_)|pval|p_?value|p_sim|p_norm", str(c).lower()):
            continue
        try:
            v = layer[c].astype(float).to_numpy()
        except Exception:
            continue
        f = v[np.isfinite(v)]
        if len(f) >= 10 and f.min() >= 0 and f.max() <= 1 and f.std() > 0:
            out.append(v)
    return out


def _points_drawn(ctx, cases, crss):
    """Fraction/number of drawn point offsets that sit on case locations (any candidate CRS)."""
    from scipy.spatial import cKDTree
    best = (0, 0.0, None)
    offs = [(t, m, np.asarray(a, float)) for t, m, a in drawn_arrays(ctx, "pred")
            if m["name"].endswith("_offsets")]
    if not offs:
        return best
    for crs in crss:
        try:
            c = cases.to_crs(crs)
            xy = np.c_[c.geometry.x, c.geometry.y]
            xy = xy[np.isfinite(xy).all(axis=1)]
            geo = c.crs.is_geographic
        except Exception:
            continue
        if not len(xy):
            continue
        tree = cKDTree(xy)
        tol = 1e-5 if geo else 1.0
        for t, m, a in offs:
            if a.ndim != 2 or a.shape[1] != 2:
                continue
            a = a[np.isfinite(a).all(axis=1)]   # NaN offsets would crash the query
            if len(a) < 20:
                continue
            d = tree.query(a)[0]
            n = int((d <= tol).sum())
            if n > best[0]:
                best = (n, n / len(a), f"{m['name']}@{t} in {crs}")
    return best


def checks(ctx):
    import geopandas as gpd
    out = []
    pngs = [f for f in ctx.pred_files("*.png")]
    names = list(dict.fromkeys(["blood_lead_hotspots.png", "transit_access.png"]
                               + [f for f in pngs if any(k in f.lower() for k in ("lead", "hot", "hex", "blood"))]))
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))

    cases = gpd.read_file(f"{ctx.inputs_dir}/High_Blood_Level_Results.shp")
    arrays = {}
    for tag, meta, load in ctx.objects("pred", "array"):
        sh = meta.get("shape", [])
        if len(sh) == 1 or (len(sh) == 2 and 1 in sh):
            arrays.setdefault(int(np.prod(sh)), []).append((f"{meta['name']}@{tag}", load))

    best_cnt = (0.0, None, 0)
    best_hs = (-1.0, None)
    hs_pass = []
    seen = set()
    for tag, meta, load in ctx.objects("pred", "gdf"):
        n = meta.get("rows", 0)
        sig = (meta.get("name"), n, tuple(meta.get("columns", [])))
        if sig in seen or n < 20 or n > 20000:
            continue
        seen.add(sig)
        try:
            g = load()
            if g.crs is None or not g.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                continue
            if _hexness(g) < 0.5:
                continue
            g = g.reset_index(drop=True)
            pts = cases.to_crs(g.crs)
            j = gpd.sjoin(pts, g[["geometry"]], how="inner", predicate="within")
            recount = np.bincount(j["index_right"].to_numpy(), minlength=len(g))
            held = int(recount.sum())
            numcols = [c for c in g.columns if c != "geometry" and g[c].dtype.kind in "fiub"]
            for c in numcols:
                v = g[c].astype(float).fillna(0).to_numpy()
                agree = float(np.mean(np.abs(v - recount) < 0.5))
                if held >= 0.9 * N_CASES and agree > best_cnt[0]:
                    best_cnt = (agree, f"{c} in {meta['name']}@{tag}", held)
            if held < 0.5 * N_CASES:
                continue
            z = _gistar_z(g, recount)
            cands = [(f"{c} in {meta['name']}@{tag}", g[c].astype(float).to_numpy()) for c in numcols]
            for nm, ld in arrays.get(len(g), []):
                try:
                    cands.append((nm, np.asarray(ld(), float).ravel()))
                except Exception:
                    pass
            for nm, v in cands:
                if not (_spearman(v, recount) < 0.97):
                    continue
                rho = _spearman(v, z)
                if rho == rho and rho >= 0.6 and _zlike(v, z):
                    dr = _drawn_stat(ctx, g, v) if len(v) == len(g) else None
                    hs_pass.append((rho, nm, dr))
                if rho == rho and rho > best_hs[0]:
                    best_hs = (rho, nm)
        except Exception:
            continue
    out.append(result("hexagon tessellation with correct case counts", best_cnt[0] >= 0.95,
                      value={"cells_with_exact_count": round(best_cnt[0], 3), "cases_in_cells": best_cnt[2]},
                      threshold=">=0.95 of cells exact, >=90 % of 373 cases covered",
                      note=best_cnt[1] or "no hexagon layer with a matching count column"))
    hs = max(hs_pass, key=lambda x: x[0]) if hs_pass else None
    out.append(result("hot spot statistic consistent with Gi* on the hexagon counts", hs is not None,
                      value=round(hs[0], 3) if hs else round(best_hs[0], 3),
                      threshold="Spearman >= 0.6 with recomputed Gi* z and Gi*-type values (signed z or bins)",
                      note=(hs[1] if hs else (f"best {best_hs[1]} is not a Gi*-type product (no sign / scale)"
                                              if best_hs[1] else "no candidate column"))))
    drawn = [x for x in hs_pass if x[2]]
    out.append(result("the map draws the hot spot result on the hexagons", bool(drawn),
                      value=drawn[0][2] if drawn else None,
                      note=drawn[0][1] if drawn else "no drawn hexagon layer shows a passing hot spot column"))
    try:
        crss = {str(cases.crs), "EPSG:4326", "EPSG:3857"}
        for tag, meta, load in ctx.objects("pred", "gdf", include_files=False):
            if meta.get("crs") and meta["crs"] != "None":
                crss.add(meta["crs"])
        n, frac, where = _points_drawn(ctx, cases, sorted(crss))
    except Exception as e:
        n, frac, where = 0, 0.0, "error " + repr(e)[:80]
    out.append(result("privacy: individual case locations are not drawn", n < 20, value=n,
                      threshold="< 20 drawn points on case locations", note=where or "no case points in the figure"))

    code = ctx.code_text()
    import re
    fid = bool(re.search(r"G_Local\s*\(|getisord\.\w+\s*\(|Getis|gi_star|G_star|\bGi\*", code))
    out.append(result("method fidelity: Getis-Ord hot spot code", fid, required=False))
    try:
        areas = []
        for tag, meta, load in ctx.objects("pred", "gdf"):
            if meta.get("rows", 0) >= 20:
                g = load()
                if g.crs is not None and g.geom_type.isin(["Polygon"]).all() and _hexness(g) >= 0.5:
                    areas.append(float(np.median(g.to_crs(3310).area)) / 1e6)
        a = min(areas) if areas else float("nan")
        out.append(result("privacy: median hexagon area >= 0.05 km2", a >= 0.05, value=round(a, 3) if a == a else None,
                          threshold=0.05, required=False))
    except Exception as e:
        out.append(result("privacy: median hexagon area >= 0.05 km2", False, required=False, note=repr(e)[:100]))
    return out
