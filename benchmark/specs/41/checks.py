"""Checks for task 41: campsite data quality (Wyoming) - join, relate, relationship class.

The spatial layers are supplied as GeoPackages, so all three parts
of the instruction are judged. Gold = reference.py (keys from the .lyrx files).

Correctness (required):
  1. some result file written to pred_results;
  2. relate Campsites.CampsiteID -> CampsiteAttributes.EntityID materialised per campsite: a table
     keyed by campsite id, wide (one row per campsite: a count column, or the per-row non-null count
     of a value pivot) or long (repeated keys with a per-campsite constant column), equal to the gold
     number of related records / distinct attribute names / valued attribute names for >= 95 % of
     the 1,170 Wyoming campsites. A table over all US campsites passes as well (it contains them).
     Captured objects (function locals / returns included) and table files in pred_results
     (*.csv read directly below 2 GB, *.xlsx, *.gpkg, *.shp, *.geojson) are examined, for 2-4 alike;
  3. park-level join National Parks.ParentOrgID -> OrganizationsTable.OrgID: a table whose rows for
     both parks (Yellowstone, Grand Teton) carry the organisation 'National Park Service' (OrgName,
     or OrgAbbrevName 'NPS') and not 'Historic Hotels of America' (what the OrgParentID key gives);
  4. relationship campground (Facilities.FacilityID) 1 -> many Campsites (ParentFacilityID)
     materialised, in either form:
     - per campground: a table keyed by FacilityID or FacilityName whose count column equals the
       gold number of campsites for every facility that has campsites (6), or
     - per campsite: a table keyed by CampsiteID covering >= 95 % of the campsites whose parent
       facility exists, with the parent's FacilityName (or the joined FacilityID) equal to gold for
       >= 95 % of them, and no real facility (name or id) for >= 95 % of the campsites without a parent
       facility that the table holds (labels such as 'ORPHAN' or -1 are fine; assigning those 900
       campsites to some campground, e.g. by distance, fails). A per-campground table keyed by
       FacilityID must not list the 5 parent ids missing from Facilities (plain foreign-key counts
       without the Facilities table are not the relationship).
     The relate also accepts a merged long table (campsite-layer columns + attribute records) by its
     rows per campsite; value pivots are counted over the attribute-name columns only
     (names are also matched after normalisation - case, non-alphanumerics -
     and after stripping a prefix shared by the pivot block, e.g. 'attr_' + sanitised name).
"""
import os
import re

import numpy as np
import pandas as pd

from checklib import result


def _gold(ctx, name):
    for tag, meta, load in ctx.objects("gold", include_files=False):
        if meta["name"] == name:
            return load()
    return None


def _tables(ctx, min_rows):
    """(label, DataFrame) for captured objects and table files of the prediction."""
    seen = []
    for tag, meta, load in ctx.objects("pred", include_files=False):
        if meta.get("kind") not in ("df", "gdf", "series") or meta.get("rows", 0) < min_rows:
            continue
        try:
            d = load()
        except Exception:
            continue
        if isinstance(d, pd.Series):
            d = d.to_frame()
        seen.append((f"{meta['name']}@{tag}", d))
    for f in ctx.pred_files():
        p = ctx.pred(f)
        ext = f.lower().rsplit(".", 1)[-1] if "." in f else ""
        try:
            if ext == "csv" and os.path.getsize(p) < 2e9:
                d = pd.read_csv(p, low_memory=False)
            elif ext in ("xlsx", "xls"):
                d = pd.read_excel(p)
            elif ext in ("shp", "gpkg", "geojson"):
                import geopandas as gpd
                d = gpd.read_file(p)
            else:
                continue
        except Exception:
            continue
        if len(d) >= min_rows:
            seen.append((f"{f}@file", d))
    out = []
    for lab, d in seen:
        d = d.drop(columns=[c for c in d.columns if c == "geometry"])
        if not isinstance(d.index, pd.RangeIndex):   # keys kept in the index (groupby results)
            d = d.reset_index()
        d.columns = [str(c) for c in d.columns]
        out.append((lab, d))
    return out


def _num(s):
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _eq(a, b):
    """Element-wise a == b as a plain bool array; a missing value (pd.NA from nullable dtypes)
    counts as not equal instead of crashing the check."""
    return pd.array(a == b, dtype="boolean").fillna(False).to_numpy(dtype=bool)


_CACHE = {}


def _attribute_names(ctx):
    """All AttributeName values of the input table (columns of a value pivot)."""
    if "names" not in _CACHE:
        try:
            a = pd.read_csv(os.path.join(ctx.inputs_dir, "CampsiteAttributes.csv"), usecols=["AttributeName"],
                            encoding="utf-8-sig")
            _CACHE["names"] = set(a["AttributeName"].dropna().astype(str))
        except Exception:
            _CACHE["names"] = set()
    return _CACHE["names"]


def _key(s):
    """Normalised attribute/column name: lower case, runs of non-alphanumerics -> '_'."""
    return re.sub(r"[^0-9a-z]+", "_", str(s).strip().lower()).strip("_")


def _pivot_columns(ctx, columns):
    """Columns of a wide value pivot that are attribute names: exact AttributeName matches,
    and matches after normalisation (_key) and after stripping a
    prefix shared by the pivot block (e.g. 'attr_' + sanitised name). The prefix giving the most
    matches wins."""
    names = _attribute_names(ctx)
    exact = [c for c in columns if c in names]
    keys = {_key(n) for n in names} - {""}
    prefixes = {""} | {m.group(0) for c in columns for m in [re.match(r"^[A-Za-z0-9]+[^A-Za-z0-9]+", str(c))] if m}
    best = exact
    for p in prefixes:
        hit = [c for c in columns if str(c).startswith(p) and _key(str(c)[len(p):]) in keys]
        if len(hit) > len(best):
            best = hit
    return best


def _facilities(ctx):
    if "fac" not in _CACHE:
        try:
            import geopandas as gpd
            _CACHE["fac"] = gpd.read_file(os.path.join(ctx.inputs_dir, "Facilities.gpkg"))
        except Exception:
            _CACHE["fac"] = None
    return _CACHE["fac"]


def _norm(s):
    return s.astype(str).str.strip().str.lower()


def check_relate(ctx, gold, wy_ids):
    """wy_ids: all Wyoming campsites (from the campsite layer, incl. one without attribute records)."""
    name = "one-to-many campsite relate materialised (per-campsite related-record counts, Wyoming campsites)"
    gold = gold.set_index("CampsiteID")
    all_ids = set(gold.index.astype("int64")) | set(wy_ids)
    n = len(wy_ids)
    best = [None, 0.0]

    def score(lab, ids, vals):
        s = pd.Series(vals, index=ids)
        s = s[~s.index.duplicated()]
        for gcol in ("n_attributes", "n_attribute_names", "n_valued_attribute_names"):
            g = gold[gcol].reindex(wy_ids).fillna(0).values
            v = s.reindex(wy_ids).values
            frac = float(np.sum(np.isfinite(v) & (v == g)) / n)
            if frac > best[1]:
                best[0], best[1] = f"{lab} ~ {gcol}", frac

    for lab, d in _tables(ctx, int(0.9 * n)):
        key = None
        for c in d.columns:
            v = _num(d[c])
            if v.notna().mean() > 0.95 and v.dropna().astype("int64").isin(all_ids).mean() > 0.95:
                key = (c, v)
                break
        if key is None:
            continue
        kc, key = key
        m = key.notna().values
        num = [c for c in d.columns if c != kc and d[c].dtype.kind in "iuf"]
        if key[m].is_unique:
            ids = key[m].astype("int64").values
            for c in num:
                score(f"{lab}.{c}", ids, _num(d[c]).values[m])
            names = _pivot_columns(ctx, d.columns)
            if len(names) > 20:  # wide value pivot: non-null count over the attribute-name columns only
                score(f"{lab} row non-null count", ids, d[names].notna().sum(axis=1).values[m])
        elif pd.Series(key[m].astype("int64").unique()).isin(wy_ids).sum() >= 0.9 * n:
            k = key[m].astype("int64")
            low = {c.lower() for c in d.columns}
            # merged long relate table (campsite layer columns + attribute records): rows per campsite.
            # The raw attribute table alone has no campsite-layer column and is not counted.
            if low & {"campsitename", "parentfacilityid"} and low & {"attributename", "attributeid", "attributevalue"}:
                per = k.value_counts()
                score(f"{lab} rows per campsite (long merged)", per.index.values, per.values.astype(float))
            for c in num:
                g = _num(d[c])[m].groupby(k.values)
                if (g.nunique() <= 1).mean() < 0.99:
                    continue  # per-record column, not a per-campsite attribute of the long table
                per = g.first()
                score(f"{lab}.{c} (long form)", per.index.values, per.values.astype(float))
    return result(name, best[1] >= 0.95, value=round(best[1], 4), threshold=">= 0.95 of the 1,170 Wyoming campsites exact",
                  note=best[0] or "no per-campsite table keyed by CampsiteID/EntityID")


def check_join(ctx):
    name = "park-level join ParentOrgID -> OrgID (both parks carry 'National Park Service')"
    hits = []
    for lab, d in _tables(ctx, 2):
        s = d.astype(str).apply(lambda col: col.str.strip().str.lower())
        rows = {}
        for park, pats in (("YELL", ("yell", "yellowstone national park", "yellowstone")),
                           ("GRTE", ("grte", "grand teton national park", "grand teton"))):
            r = s.isin(pats).any(axis=1)
            if r.any():
                rows[park] = s[r]
        if len(rows) < 2:
            continue
        ok = all(((r == "national park service") | (r == "nps")).any(axis=1).all() for r in rows.values())
        wrong = any((r == "historic hotels of america").any(axis=1).any() for r in rows.values())
        hits.append((lab, ok and not wrong))
    good = [lab for lab, ok in hits if ok]
    return result(name, bool(good), value=good[:3] or [lab for lab, _ in hits][:3],
                  threshold="a table with both parks and their organisation",
                  note="" if hits else "no table holding both parks")


def check_relationship(ctx, counts, link):
    name = "relationship campground (FacilityID) 1 -> many campsites (ParentFacilityID) materialised"
    counts = counts[counts["n_campsites"] > 0]
    g_by_id = counts.set_index("FacilityID")["n_campsites"]
    g_by_name = counts.assign(k=_norm(counts["FacilityName"])).set_index("k")["n_campsites"]
    matched = link[link["FacilityName"].notna()]
    g_name = pd.Series(_norm(matched["FacilityName"]).values, index=matched["CampsiteID"].astype("int64").values)
    g_fid = link.set_index(link["CampsiteID"].astype("int64"))["FacilityID"]
    fac = _facilities(ctx)
    fac_ids = set(fac["FacilityID"].astype("int64")) if fac is not None else set(g_fid.dropna().astype("int64"))
    fac_names = set(_norm(fac["FacilityName"])) if fac is not None else set(g_name.values)
    orphan_fk = set(link["ParentFacilityID"].astype("int64")) - fac_ids
    best = [None, 0.0]
    for lab, d in _tables(ctx, len(counts)):
        # (a) per campground
        for kc in d.columns:
            kv = _num(d[kc])
            if kv.notna().any() and kv.isin(g_by_id.index).sum() >= len(g_by_id):
                if kv.isin(orphan_fk).any():
                    continue  # foreign-key counts without the Facilities table (no join)
                gold, key = g_by_id, kv
            else:
                kn = _norm(d[kc])
                if kn.isin(g_by_name.index).sum() >= len(g_by_name):
                    gold, key = g_by_name, kn
                else:
                    continue
            sub = d[key.isin(gold.index).values]
            ksub = key[key.isin(gold.index)]
            if not ksub.is_unique:
                continue
            for c in d.columns:
                if c == kc or d[c].dtype.kind not in "iuf":
                    continue
                v = pd.Series(_num(sub[c]).values, index=ksub.values)
                frac = float(_eq(v.reindex(gold.index).values, gold.values).mean())
                if frac > best[1]:
                    best[0], best[1] = f"{lab}: {c} by {kc} (per campground)", frac
        # (b) per campsite
        for kc in d.columns:
            kv = _num(d[kc])
            if not (kv.notna().mean() > 0.9 and kv.isin(g_name.index).sum() >= 0.95 * len(g_name)):
                continue
            m = kv.notna().values
            ids = kv[m].astype("int64")
            if not ids.is_unique:
                continue
            for c in d.columns:
                if c == kc:
                    continue
                col = d[c][m]
                if col.dtype.kind in "iuf":   # joined FacilityID: equal for matched, missing otherwise
                    v = pd.Series(_num(col).values, index=ids.values)
                    gv = g_fid.reindex(ids.values)
                    # orphans: any value that is not a real FacilityID (missing, -1, ...)
                    # (the campsite layer's own ParentFacilityID keeps the dangling key: not a join)
                    eq = _eq(v.values, gv.values) | (gv.isna().values & ~pd.Series(v.values).isin(fac_ids | orphan_fk).values)
                    covered = gv.index.isin(g_name.index)
                    frac = float(min(eq[covered].mean() if covered.any() else 0, eq.mean()))
                else:   # parent FacilityName: equal for matched campsites, missing for orphans
                    v = pd.Series(_norm(col).where(col.notna(), None).values, index=ids.values)
                    vm = v.reindex(g_name.index)
                    frac = float((vm.values == g_name.values).mean())
                    orph = v[~v.index.isin(g_name.index)]
                    if len(orph):
                        frac = min(frac, float((~orph.isin(fac_names)).mean()))  # orphans: no real facility name
                if frac > best[1]:
                    best[0], best[1] = f"{lab}: {c} by {kc} (per campsite)", frac
    return result(name, best[1] >= 0.95, value=round(best[1], 4),
                  threshold="per campground: all 6 counts exact; per campsite: >= 0.95 parent facility exact",
                  note=best[0] or "no campground-campsite table found")


def checks(ctx):
    out = []
    files = ctx.nonempty_pred_files()   # 0-byte files are not results
    out.append(result("result file written to pred_results", bool(files), value=files[:6], threshold="any non-empty file"))
    relate, counts, link = _gold(ctx, "campsite_relate"), _gold(ctx, "campground_campsites"), _gold(ctx, "campsite_campground")
    if relate is None or counts is None or link is None:
        for n in ("one-to-many campsite relate materialised", "park-level join", "relationship campground -> campsites"):
            out.append(result(n, False, note="gold missing"))
        return out
    out.append(check_relate(ctx, relate, np.array(sorted(link["CampsiteID"].astype("int64")))))
    out.append(check_join(ctx))
    out.append(check_relationship(ctx, counts, link))
    return out
