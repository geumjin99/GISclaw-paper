"""T42 - Local Moran's I (LISA) of Airbnb prices in Berlin.

Task core: join 20 053 listings to the 140 neighbourhood polygons, take the MEDIAN price per
neighbourhood_group (12 districts), give it to every neighbourhood of the group, fill the two
neighbourhoods without a group, build Queen contiguity weights and run Local Moran's I; map the
significant hot (HH) and cold (LL) spots to moran_local.png.

Correctness (required):
  1. moran_local.png exists and is not blank;
  2. the analysed variable is the group MEDIAN price: some captured table with one row per
     neighbourhood (138-140 rows, aligned on row order / geometry) has a numeric column equal
     to the gold group medians (within 3 %) for >= 90 % of neighbourhoods (the 2 rows without a
     group may be imputed differently or dropped). Group means, per-
     neighbourhood medians or raw listing prices fail;
  3. missing data handled sensibly: that column has no NaN and every value lies inside the range
     of observed group medians (+/-10 %) - filling with 0 fails;
  4. LISA clusters agree with gold (and no captured label that
     matches the gold hot set says "cold"/"LL" or vice versa; esda codes 3/1 or signs inverted fail): among captured per-neighbourhood categorical results (table
     column or length-140 array, any encoding: q codes, labels, booleans) the best-matching
     category reproduces the gold significant hot-spot set and cold-spot set with IoU >= 0.7
     each. Different permutation seeds/counts or analytical p-values stay well above this.
  5. Local Moran's I itself (the task names it; Getis-Ord Gi* does not count): a captured
     per-neighbourhood vector with |r| >= 0.95 to local I of the gold medians, or a
     Moran_Local call in the code;
  Medians held only in 1-D numpy arrays are accepted for check 2.
Check 3 passes when ANY median column matching gold (>= 0.90) is NaN-free and
plausible (a NaN intermediate table does not mask the filled one); in check 4 a gold hot or
cold set labelled as a spatial outlier (HL / LH, doughnut / diamond) counts as mislabelled.
Check 4 also scores quadrant and significance held
in two columns of one table (quadrant labels HH/LL/... or codes 1-4, combined with a boolean /
0-1 'sig' column or a p column thresholded at 0.05: quadrant where significant, else 'ns'), and a
quadrant array variable (q = lisa.q) combined with a p-value array variable of the same snapshot
(not the esda object's own attributes, which every Moran_Local run holds).
Columns whose labels already claim spots/clusters ('Hot spot') are not combined.
  6. Queen contiguity weights (required; the domain knowledge prescribes
     them for the spatial-similarity step): a captured libpysal Queen weights object, Queen in
     the code, or polygon neighbours hand-built by touches/intersects; Rook / KNN / distance
     band fail.
Presentation (kind="figure"; secondary analysis, never required):
  7. on the saved map the gold hot spots are drawn warmer (red - blue) than the gold cold spots
     (a map with the colours swapped while the table is right fails). UNVERIFIABLE when no per-patch colours were
     captured (e.g. hot and cold drawn as separate single-colour layers).
"""
import numpy as np

from checklib import file_exists, png_not_blank, result, find_object, key_values

N = 140
NAMES = ["moran_local.png", "Moran_Local.png", "moran_local.PNG"]


def _inputs(ctx):
    import geopandas as gpd
    g = gpd.read_file(f"{ctx.inputs_dir}/berlin-neighbourhoods.geojson")
    g["_rid"] = np.arange(len(g))
    return g


def _gold(ctx):
    obj, _, _ = find_object(ctx, "gold", "gdf", lambda o, m: len(o) == N and "lisa_cluster" in o.columns)
    return obj


def _tables(ctx):
    """Captured pred tables with 138-140 rows, with their row -> _rid alignment."""
    out = []
    g = _inputs(ctx)
    for tag, meta, load in ctx.objects("pred"):
        if meta.get("kind") not in ("gdf", "df") or not (N - 2 <= meta.get("rows", 0) <= N):
            continue
        try:
            obj = load()
            keys = key_values(obj, g, "_rid")
            if keys is None and meta.get("kind") == "gdf" and len(obj) == N:
                keys = obj.index.to_series()
            if keys is None:
                continue
            out.append((meta["name"] + "@" + tag, obj, np.asarray(keys)))
        except Exception:
            continue
    return out


def _arrays(ctx, g_valid):
    """Captured 1-D arrays of length 140 (or 138 = rows with a group)."""
    out = []
    for tag, meta, load in ctx.objects("pred", "array"):
        shp = meta.get("shape") or []
        if len(shp) != 1 or shp[0] not in (N, int(g_valid.sum())):
            continue
        try:
            a = load()
        except Exception:
            continue
        rid = np.arange(N) if len(a) == N else np.flatnonzero(g_valid)
        out.append((meta["name"] + "@" + tag, a, rid))
    return out


def _num(x):
    try:
        return float(x)
    except Exception:
        return None


def _polarity(cat, col):
    """'hot' / 'cold' / None from a category label (or, for True/1 flags, the column name)."""
    import re
    t = str(cat).lower()
    if t in ("true", "1", "1.0") and not re.search(r"[a-z]", t):
        pass
    if t in ("true",):
        t = str(col).lower()
    hot = bool(re.search(r"hot|high[-_ ]?high|\bhh\b", t))
    cold = bool(re.search(r"cold|low[-_ ]?low|\bll\b", t))
    # spatial-outlier labels (HL / LH) conflict with a hot or cold spot
    outl = bool(re.search(r"high[-_ ]?low|low[-_ ]?high|\bhl\b|\blh\b|doughnut|diamond|outlier", t))
    if outl and not hot and not cold:
        return "outlier"
    return "hot" if hot and not cold else "cold" if cold and not hot else None


def _local_moran(ctx, y):
    """Local I (row-standardised Queen, as esda) of the gold group medians on the 140 polygons."""
    import libpysal
    g = _inputs(ctx)
    w = libpysal.weights.Queen.from_dataframe(g, use_index=False, silence_warnings=True)
    w.transform = "r"
    z = (y - y.mean()) / y.std()
    lag = libpysal.weights.lag_spatial(w, z)
    return z * lag


_P_NAME = r"(^|[_.])p($|_)|pval|p_?value|p_sim|p_norm|p_z_sim|pvalue"
_SIG_NAME = r"sig"


def _pure_quadrant(v):
    """A LISA quadrant column/array: 2-4 distinct non-null values that are quadrant codes (1-4)
    or quadrant labels (HH / High-High ...). Labels that already claim a cluster or spot
    ('Hot spot', 'HH cluster', 'Not significant') are not quadrants: such a column states its own
    significance and is scored as it stands (non-significant cells labelled 'Hot spot' stay wrong)."""
    import re
    u = [x for x in np.unique(np.asarray(v).astype(str)) if x not in ("nan", "None", "")]
    if not 2 <= len(u) <= 4:
        return False
    nums = [_num(x) for x in u]
    if all(n is not None for n in nums):
        return all(n in (1.0, 2.0, 3.0, 4.0) for n in nums)
    for x in u:
        t = x.lower()
        if re.search(r"spot|cluster|sig|outlier|none|^ns$|not", t):
            return False
        if not re.fullmatch(r"\s*(hh|ll|hl|lh|high[-_ ]?high|low[-_ ]?low|high[-_ ]?low|low[-_ ]?high)\s*", t):
            return False
    return True


def _sig_flags(obj):
    """(name, bool array) significance flags of a table: boolean columns, 0/1 columns named
    like 'sig', and p-value columns (values in [0, 1]) thresholded at 0.05."""
    import re
    out = []
    for c in obj.columns:
        if c == "geometry":
            continue
        s = obj[c]
        try:
            if s.dtype.kind == "b":
                out.append((str(c), s.to_numpy(bool)))
                continue
            if s.dtype.kind not in "fiu":
                continue
            v = s.to_numpy(float)
        except Exception:
            continue
        f = v[np.isfinite(v)]
        if not len(f):
            continue
        if re.search(_SIG_NAME, str(c), re.I) and set(np.unique(f)) <= {0.0, 1.0}:
            out.append((str(c), np.nan_to_num(v) == 1))
        elif re.search(_P_NAME, str(c), re.I) and f.min() >= 0 and f.max() <= 1 and f.std() > 0:
            out.append((str(c), np.where(np.isfinite(v), v, 1.0) <= 0.05))
    return out


def _quad_sig_pairs(obj):
    """(quadrant column, labels, significance column, flags) pairs within one table."""
    flags = _sig_flags(obj)
    if not flags:
        return []
    out = []
    for c in obj.columns:
        if c in ("geometry", "neighbourhood", "neighbourhood_group", "_rid", "index", "index_right"):
            continue
        s = obj[c]
        if s.dtype.kind == "b" or not _pure_quadrant(s.to_numpy()):
            continue
        for sc, sv in flags:
            if sc != str(c):
                out.append((str(c), s.astype(str).to_numpy(), sc, sv))
    return out


def _quad_p_arrays(ctx, g_valid):
    """Quadrant arrays (codes 1-4) that the agent extracted into its own variables (e.g.
    q = lisa.q) combined with a p-value array variable of the same snapshot and length (name like
    p_sim): q where p <= 0.05, else 'ns'. Attributes of the esda object itself (captured
    automatically as 'lisa.q', 'lisa.p_sim') are NOT combined: every Moran_Local call holds both,
    so pairing them would credit any run that called Moran_Local, whatever it labelled as hot/cold
    (control wrong_nonsig_quadrants_labelled_hotcold)."""
    import re
    by = {}
    for tag, meta, load in ctx.objects("pred", "array"):
        shp = meta.get("shape") or []
        if len(shp) != 1 or shp[0] not in (N, int(g_valid.sum())):
            continue
        if "." in str(meta.get("name", "")):
            continue                     # object attribute (lisa.q / lisa.p_sim), see docstring
        by.setdefault((tag, shp[0]), []).append((meta["name"], load))
    out = []
    for (tag, n), items in by.items():
        qs, ps = [], []
        for nm, load in items:
            try:
                a = np.asarray(load())
            except Exception:
                continue
            if a.dtype.kind in "iuf" and _pure_quadrant(a) and np.isfinite(a.astype(float)).all():
                qs.append((nm, a))
            elif a.dtype.kind == "f" and re.search(_P_NAME, nm, re.I):
                f = a[np.isfinite(a)]
                if len(f) and f.min() >= 0 and f.max() <= 1 and f.std() > 0:
                    ps.append((nm, np.where(np.isfinite(a), a, 1.0) <= 0.05))
        rid = np.arange(N) if n == N else np.flatnonzero(g_valid)
        for qn, q in qs:
            for pn, sig in ps:
                out.append((f"{qn}|{pn}@{tag}", np.where(sig, q.astype(int).astype(str), "ns"), rid))
    return out


def _iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def _map_colours(ctx, fname, ghot, gcold):
    """(Presentation) Hot spots drawn in warmer colours than cold spots on the
    saved map (geopandas sorts categories alphabetically, so a hand-made colour list can land
    on the wrong classes). Every drawn polygon collection of the figure that ended up in `fname`
    is matched back to neighbourhoods by the bounding-box centres of its patches (in the CRS of a
    drawn GeoDataFrame layer or EPSG:4326, within 5 % of the typical neighbourhood size); a
    neighbourhood's colour is its visible face colour (edge colour when the face is transparent)
    in the last collection that covers it. Warmth = red - blue. Passes when the mean warmth of
    the gold hot spots exceeds that of the gold cold spots by >= 0.2. Returns
    (passed, value, note) or None when no per-patch colours were captured (single-colour layers
    leave none)."""
    import re
    from scipy.spatial import cKDTree
    from checklib import drawn_arrays, drawn_geo_layers
    by = {}
    for tag, meta, a in drawn_arrays(ctx, "pred", file=fname):
        m = re.match(r"(fig_ax(\d+)_coll(\d+))_(paths_bbox|facecolors|edgecolors)$", meta.get("name", ""))
        if m:
            by.setdefault((tag, int(m.group(2)), int(m.group(3))), {})[m.group(4)] = np.asarray(a, float)
    if not by:
        return None
    g = _inputs(ctx)
    crss = ["EPSG:4326"] + [r.get("crs") for _, r, _ in drawn_geo_layers(ctx, "pred", file=fname) if r.get("crs")]
    geo = {}
    for c in dict.fromkeys(crss):
        try:
            gg = g.to_crs(c)
            pc, own = [], []
            for i, geom in enumerate(gg.geometry):
                for part in (geom.geoms if hasattr(geom, "geoms") else [geom]):
                    b = part.bounds
                    pc.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
                    own.append(i)
            geo[c] = (cKDTree(np.asarray(pc)), np.asarray(own), float(np.sqrt(np.median(gg.geometry.area))))
        except Exception:
            continue
    colour = {}
    used = []
    for key in sorted(by):                     # drawing order within the figure: later on top
        d = by[key]
        bb = d.get("paths_bbox")
        if bb is None or bb.ndim != 2 or bb.shape[1] != 4:
            continue
        cen = np.c_[(bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2]
        if not np.isfinite(cen).all():
            continue
        best = None
        for c, (tree, own, size) in geo.items():
            dist, j = tree.query(cen)
            ok = dist <= 0.05 * size
            if best is None or ok.mean() > best[0]:
                best = (ok.mean(), ok, own[j])
        if best is None or best[0] < 0.8:
            continue
        fc, ec = d.get("facecolors"), d.get("edgecolors")
        n = len(bb)
        if fc is None or fc.ndim != 2 or len(fc) != n:
            continue
        for k in np.flatnonzero(best[1]):
            rgba = fc[k]
            if rgba.shape[0] >= 4 and rgba[3] < 0.05:
                if ec is not None and ec.ndim == 2 and len(ec) == n and ec[k][3] >= 0.05:
                    rgba = ec[k]
                else:
                    continue
            colour[int(best[2][k])] = rgba[:3]
        used.append(f"coll{key[2]}@{key[0]}")
    hot = [colour[i] for i in np.flatnonzero(ghot) if i in colour]
    cold = [colour[i] for i in np.flatnonzero(gcold) if i in colour]
    if not hot or not cold:
        return None
    wh = float(np.mean([c[0] - c[2] for c in hot]))
    wc = float(np.mean([c[0] - c[2] for c in cold]))
    return wh - wc >= 0.2, {"warmth_hot": round(wh, 3), "warmth_cold": round(wc, 3)}, ", ".join(used)


def checks(ctx):
    out = []
    r = file_exists(ctx, NAMES)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("map not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("map not blank", False, note="no map"))

    gold = _gold(ctx)
    if gold is None:
        for n in ("analysed variable = group median price", "missing values handled (no NaN, plausible fill)",
                  "LISA hot/cold spots match gold (IoU >= 0.7)"):
            out.append(result(n, False, note="gold table missing"))
        return out
    gmed = gold["median_pri"].to_numpy(float)
    has_grp = gold["neighbourhood_group"].notna().to_numpy()
    lo, hi = gmed[has_grp].min(), gmed[has_grp].max()

    tables = _tables(ctx)
    # --- 2/3: median price column
    best = (None, -1.0, None)
    cands_med = []      # every (name, frac, values), see check 3
    for name, obj, keys in tables:
        ok_k = (keys >= 0) & (keys < N) if keys.dtype.kind in "iuf" else None
        if ok_k is None:
            continue
        for c in obj.columns:
            if c in ("geometry", "_rid", "index", "index_right") or obj[c].dtype.kind not in "fi":
                continue
            v = obj[c].to_numpy(float)
            k = keys[ok_k].astype(int)
            vv = v[ok_k]
            close = np.abs(vv - gmed[k]) <= 0.03 * gmed[k]
            frac = float(close[has_grp[k]].sum() / has_grp.sum())
            cands_med.append((f"{c} in {name}", frac, vv))
            if frac > best[1]:
                best = (f"{c} in {name}", frac, vv)
    for name, a, rid in _arrays(ctx, has_grp):     # values held only in 1-D arrays
        if a.dtype.kind not in "fiu":
            continue
        v = a.astype(float)
        close = np.abs(v - gmed[rid]) <= 0.03 * gmed[rid]
        frac = float(close[has_grp[rid]].sum() / has_grp.sum())
        cands_med.append((name, frac, v))
        if frac > best[1]:
            best = (name, frac, v)
    out.append(result("analysed variable = group median price", best[1] >= 0.9,
                      value=round(best[1], 3), threshold=">=0.90 of neighbourhoods within 3 % of gold",
                      note=best[0] or "no per-neighbourhood table"))
    if best[0] and best[1] >= 0.9:
        # an intermediate table that still holds NaN (captured before the filled / dropped one)
        # does not decide this check: any matching (>= 0.90) median column without NaN and with
        # plausible values passes; the reported values are those of the best such column.
        def _ok(v):
            return bool(np.isfinite(v).all() and (v >= 0.9 * lo).all() and (v <= 1.1 * hi).all())
        good = [(n_, f_, v_) for n_, f_, v_ in cands_med if f_ >= 0.9 and _ok(v_)]
        pick = max(good, key=lambda t: t[1]) if good else best
        v = pick[2]
        ok = bool(good)
        out.append(result("missing values handled (no NaN, plausible fill)", ok,
                          value=[float(np.nanmin(v)), float(np.nanmax(v)), int(np.isnan(v).sum())],
                          threshold=f"[{0.9 * lo:.1f}, {1.1 * hi:.1f}], no NaN", note=pick[0]))
    else:
        out.append(result("missing values handled (no NaN, plausible fill)", False, note="no median column"))

    # --- 4: LISA clusters
    ghot = gold["lisa_cluster"].to_numpy() == 1
    gcold = gold["lisa_cluster"].to_numpy() == 3
    cands = []
    for name, obj, keys in tables:
        if keys.dtype.kind not in "iuf":
            continue
        k = keys.astype(int)
        if len(np.unique(k)) != len(k) or (k < 0).any() or (k >= N).any():
            continue
        for c in obj.columns:
            if c in ("geometry", "neighbourhood", "neighbourhood_group", "_rid", "index", "index_right"):
                continue
            s = obj[c]
            if s.dtype.kind == "f" and s.nunique() > 8:
                continue
            if s.nunique(dropna=False) < 2 or s.nunique(dropna=False) > 8:
                continue
            cands.append((f"{c} in {name}", s.astype(str).to_numpy(), k))
        # quadrant and significance held in two
        # columns -> also score quadrant.where(significant, 'ns')
        for qc, qv, sc, sv in _quad_sig_pairs(obj):
            cands.append((f"{qc}|{sc} in {name}", np.where(sv, qv, "ns"), k))
    for name, a, rid in _arrays(ctx, has_grp):
        if a.dtype.kind in "biu" or (a.dtype.kind == "f" and len(np.unique(a)) <= 8):
            if 2 <= len(np.unique(a)) <= 8:
                cands.append((name, a.astype(str), rid))
    for name, a, rid in _quad_p_arrays(ctx, has_grp):   # captured q array + p_sim array
        cands.append((name, a, rid))
    bh, bc = (-1.0, None), (-1.0, None)
    inverted = []
    for name, a, k in cands:
        gh, gc = ghot[k], gcold[k]
        hot_cat = cold_cat = None
        for cat in np.unique(a):
            m = a == cat
            ih, ic = _iou(m, gh), _iou(m, gc)
            if ih > bh[0]:
                bh = (ih, f"{name}=={cat}")
            if ic > bc[0]:
                bc = (ic, f"{name}=={cat}")
            if ih >= 0.7:
                hot_cat = cat
            if ic >= 0.7:
                cold_cat = cat
        # labels must not say the opposite (e.g. hot spots labelled "Cold spot")
        col = name.split(" in ")[0]
        if hot_cat is not None and _polarity(hot_cat, col) in ("cold", "outlier"):
            inverted.append(f"{name}: gold hot spots labelled {hot_cat!r}")
        if cold_cat is not None and _polarity(cold_cat, col) in ("hot", "outlier"):
            inverted.append(f"{name}: gold cold spots labelled {cold_cat!r}")
        if hot_cat is not None and cold_cat is not None:
            h, c = _num(hot_cat), _num(cold_cat)
            if h is not None and c is not None and ((h, c) == (3.0, 1.0) or (h < 0 < c)):
                inverted.append(f"{name}: numeric codes hot={hot_cat} cold={cold_cat} (esda q / sign inverted)")
    out.append(result("LISA hot/cold spots match gold (IoU >= 0.7), labels not inverted",
                      bh[0] >= 0.7 and bc[0] >= 0.7 and not inverted,
                      value=[round(bh[0], 3), round(bc[0], 3)], threshold=">=0.7 each, no inverted labels",
                      note=f"hot: {bh[1]}; cold: {bc[1]}; gold n_hot={int(ghot.sum())} n_cold={int(gcold.sum())}"
                           + ("; INVERTED: " + "; ".join(inverted[:3]) if inverted else "")))

    # --- 5: Local Moran's I (the statistic the task names); Gi* does not count
    li_ok, li_note = False, ""
    try:
        Is = _local_moran(ctx, gmed)
        for name, obj, keys in tables:
            if keys.dtype.kind not in "iuf":
                continue
            k = keys.astype(int)
            okk = (k >= 0) & (k < N)
            for c in obj.columns:
                if c in ("geometry", "_rid", "index", "index_right") or obj[c].dtype.kind not in "fi":
                    continue
                v = obj[c].to_numpy(float)[okk]
                m = np.isfinite(v)
                if m.sum() >= 120 and np.std(v[m]) > 0 and abs(np.corrcoef(v[m], Is[k[okk]][m])[0, 1]) >= 0.95:
                    li_ok, li_note = True, f"local I values in {c} of {name}"
        if not li_ok:
            for name, a, rid in _arrays(ctx, has_grp):
                v = a.astype(float)
                m = np.isfinite(v)
                if m.sum() >= 120 and np.std(v[m]) > 0 and abs(np.corrcoef(v[m], Is[rid][m])[0, 1]) >= 0.95:
                    li_ok, li_note = True, f"local I values in {name}"
    except Exception as e:
        li_note = repr(e)[:120]
    if not li_ok:
        import re
        code = ctx.code_text()
        if re.search(r"Moran_Local(_BV)?\s*\(|moran_local\s*\(|esda\.Moran_Local|lisa\s*\(", code):
            li_ok, li_note = True, "Moran_Local called in code"
    out.append(result("Local Moran's I computed (local I values captured or Moran_Local called)", li_ok,
                      note=li_note or "no local I values and no Moran_Local call (e.g. Gi* used instead)"))

    # --- 7: presentation (kind="figure"): hot/cold colour direction on the map
    fn = "map colours: hot spots drawn warmer than cold spots (presentation)"
    try:
        mc = _map_colours(ctx, r["value"][0] if r["passed"] else NAMES[0], ghot, gcold)
    except Exception as e:
        mc = (False, None, repr(e)[:150])
    if mc is None:
        out.append(result(fn, False, kind="figure",
                          note="UNVERIFIABLE: no per-patch colours of the neighbourhoods captured in the map"))
        out[-1]["unverifiable"] = True
    else:
        out.append(result(fn, mc[0], value=mc[1], threshold="warmth(hot) - warmth(cold) >= 0.2 (warmth = R - B)",
                          kind="figure", note=mc[2]))

    # --- 6: Queen contiguity (required: the domain knowledge prescribes
    # it for the "spatial similarity" step). Evidence: a captured weights object of type Queen
    # (also when it was built in a block that later failed, which the code text leaves out), or
    # Queen in the code, or hand-built polygon neighbours by touches/intersects (= queen rule).
    import re
    code = _strip_comments(ctx.code_text())
    q_ok, q_note = False, ""
    try:
        for tag, meta, load in ctx.objects("pred", kind="object", include_files=False):
            if re.search(r"\bQueen$", str(meta.get("object_type", ""))):
                q_ok, q_note = True, f"Queen weights object {meta['name']}@{tag}"
                break
    except Exception as e:
        q_note = repr(e)[:120]
    if not q_ok and re.search(r"queen", code, re.I):
        q_ok, q_note = True, "Queen in code"
    if not q_ok and re.search(r"\.(touches|intersects)\s*\(|predicate\s*=\s*['\"](touches|intersects)", code) \
            and re.search(r"\bW\s*\(|neighbo", code):
        q_ok, q_note = True, "polygon neighbours built by touches/intersects (queen rule)"
    out.append(result("method: Queen contiguity weights (prescribed)", q_ok,
                      note=q_note or "no Queen weights object, no Queen in code (Rook / KNN / distance band used?)"))
    return out


def _strip_comments(code):
    """Code text without comments (a method named only in a comment is not used)."""
    import io
    import tokenize
    try:
        return tokenize.untokenize(t for t in tokenize.generate_tokens(io.StringIO(code).readline)
                                   if t.type != tokenize.COMMENT)
    except Exception:
        return "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))
