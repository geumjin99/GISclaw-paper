#!/usr/bin/env python3
"""Check that each task's declarative deliverable contract (benchmark/specs/<id>/contract.json,
scoring side only) is consistent with its checks.py and its gold run. No container needed.

    python3 -W ignore harness/validate_contracts.py --tasks 1-50

Per task (errors make the task FAIL):
  schema      fields and kinds; crs_required is true exactly for vector/raster deliverables
  static      every literal file pattern of the contract appears in checks.py; every literal
              output file name in checks.py is covered by a deliverable or by not_deliverables
  gold        each required deliverable exists in the gold outputs; geographic gold files carry a
              CRS; vector geometry types and declared value ranges hold on the gold files;
              every name in checked_by is a check that checks.py returns
  ablation    gold passes; removing the gold files of a required deliverable makes it fail,
              removing an optional one does not; every crs_required gold file rewritten
              without CRS (nocrs.write_without_crs) makes the uniform CRS check fail
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import ast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
import checklib  # noqa: E402
from build_gold import evaluate  # noqa: E402
from nocrs import write_without_crs  # noqa: E402

KINDS = {"vector", "raster", "table", "figure", "text"}
GEO = {"vector", "raster"}
OUT_EXT = r"png|jpe?g|tiff?|geojson|shp|gpkg|json|csv|txt|xlsx|html|pdf|svg"
LITERAL = re.compile(r"^[\w\-. *]+\.(" + OUT_EXT + r")$", re.I)


def parse_tasks(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def correct(res):
    v = checklib.verdict(res)
    if v is None:   # a scorer crash is an error of the task's checks, not a verdict
        raise RuntimeError("checks.py crashed: " + next(c["note"] for c in res if checklib.has_scorer_error([c]))[-400:])
    return v


def gold_paths(t):
    g = os.path.join(paths.GOLD, str(t))
    return os.path.join(g, "pred_results"), os.path.join(g, "_task_root", "_snap"), g


def files_of(names, patterns):
    """Names matching patterns, plus the sidecars of matched shapefiles."""
    hit = {n for n in names if checklib.matches(n, patterns)}
    for n in list(hit):
        if n.lower().endswith(".shp"):
            stem = n[:-4].lower()
            hit |= {m for m in names if m.lower().startswith(stem + ".") and m.lower()[len(stem):] in checklib.SHP_PARTS}
    return hit


def view_of(src, drop=(), nocrs=()):
    """Temp copy of src (symlinks; real copies for files to be rewritten) without `drop`,
    with the files in `nocrs` rewritten without CRS."""
    v = tempfile.mkdtemp(prefix="contract_")
    names = os.listdir(src)
    rewrite = set()
    for n in nocrs:
        rewrite |= files_of(names, [n])
    for n in names:
        if n in drop:
            continue
        s, d = os.path.join(src, n), os.path.join(v, n)
        if n in rewrite:
            shutil.copy2(s, d)
        else:
            os.symlink(s, d)
    for n in nocrs:
        write_without_crs(os.path.join(v, n))
    return v


def literal_names(src_text, inputs):
    out = set()
    for node in ast.walk(ast.parse(src_text)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and LITERAL.match(node.value.strip()):
            v = node.value.strip()
            if "*" not in v and os.path.basename(v) not in inputs and not v.startswith("."):
                out.add(v)
    return out


def validate(t):
    errs, notes = [], []
    spec = os.path.join(paths.SPECS, str(t))
    c = checklib.load_contract(t)
    if c is None:
        return ["no contract.json (or unreadable)"], notes
    # ---- schema
    if c.get("task") != t:
        errs.append(f"task field {c.get('task')!r} != {t}")
    dels = c.get("deliverables")
    if not isinstance(dels, list) or not dels:
        return errs + ["deliverables missing/empty"], notes
    ids = set()
    for d in dels:
        i = d.get("id")
        if not i or i in ids:
            errs.append(f"deliverable id missing/duplicate: {i!r}")
        ids.add(i)
        if d.get("kind") not in KINDS:
            errs.append(f"{i}: kind {d.get('kind')!r} not in {sorted(KINDS)}")
        if not d.get("patterns") or not all(isinstance(p, str) for p in d["patterns"]):
            errs.append(f"{i}: patterns missing")
        if not isinstance(d.get("required"), bool):
            errs.append(f"{i}: required must be true/false")
        if bool(d.get("crs_required")) != (d.get("kind") in GEO):
            errs.append(f"{i}: crs_required must be true exactly for vector/raster deliverables")
        if d.get("geometry_type") and d.get("kind") != "vector":
            errs.append(f"{i}: geometry_type only for vector deliverables")
        if not d.get("checked_by"):
            errs.append(f"{i}: checked_by (names of the checks that use it) missing")
    if errs:
        return errs, notes
    # ---- static
    src = open(os.path.join(spec, "checks.py")).read()
    low = src.lower()
    inputs = set(os.listdir(os.path.join(paths.INPUTS, str(t), "dataset")))
    # ctx.objects(...) loads output files too (checklib.Ctx.file_objects) unless include_files=False
    uses_file_objects = "file_objects" in src or bool(re.search(r"\.objects\((?![^)]*include_files\s*=\s*False)", src))
    loaded_ext = {"csv", "txt", "xlsx", "xls", "shp", "geojson", "gpkg", "json", "tif", "tiff"} if uses_file_objects else set()
    for d in dels:
        for p in d["patterns"]:
            if "*" in p or "?" in p:
                ext = p.rsplit(".", 1)[-1].lower().rstrip("*")
                if ext and ext != "*" and ext not in loaded_ext and \
                        not any(f in low for f in ("." + ext, f'"{ext}"', f"'{ext}'")):
                    errs.append(f"{d['id']}: wildcard {p!r} but checks.py never mentions '.{ext}'")
            elif p.lower() not in low:
                errs.append(f"{d['id']}: file name {p!r} does not appear in checks.py")
    covered = [p for d in dels for p in d["patterns"]] + [x["pattern"] for x in c.get("not_deliverables", [])]
    for n in sorted(literal_names(src, inputs)):
        if not checklib.matches(os.path.basename(n), covered):
            errs.append(f"checks.py names output {n!r}, not covered by any deliverable / not_deliverables")
    # ---- gold
    pred, snap, groot = gold_paths(t)
    names = sorted(os.listdir(pred))
    for d in dels:
        hit = sorted(n for n in names if checklib.matches(n, d["patterns"]))
        if d["required"] and not hit:
            errs.append(f"{d['id']}: required, but no gold output matches {d['patterns']}")
        for n in hit:
            p = os.path.join(pred, n)
            if d["kind"] in GEO:
                kind, crs, note = checklib.geo_file_crs(p)
                if kind is None:
                    continue
                if kind != d["kind"]:
                    errs.append(f"{d['id']}: gold {n} is {kind}, contract says {d['kind']}")
                if not crs:
                    errs.append(f"{d['id']}: gold {n} has no CRS ({note})")
                if kind == "vector" and d.get("geometry_type"):
                    import geopandas as gpd
                    gt = set(gpd.read_file(p).geom_type.dropna().unique())
                    if not gt <= set(d["geometry_type"]):
                        errs.append(f"{d['id']}: gold {n} geometry types {sorted(gt)} not in {d['geometry_type']}")
            for col, (lo, hi) in (d.get("value_ranges") or {}).items():
                try:
                    import numpy as np
                    if d["kind"] == "raster":
                        a = checklib.read_raster(p)[0]
                        v = a[np.isfinite(a)]
                    elif d["kind"] == "vector":
                        import geopandas as gpd
                        v = gpd.read_file(p)[col].astype(float).dropna().to_numpy()
                    else:
                        import pandas as pd
                        v = pd.read_csv(p, sep=None, engine="python")[col].astype(float).dropna().to_numpy()
                    if len(v) and (v.min() < lo or v.max() > hi):
                        errs.append(f"{d['id']}: gold {n}[{col}] range [{v.min():.4g}, {v.max():.4g}] outside [{lo}, {hi}]")
                except Exception as e:
                    errs.append(f"{d['id']}: value_ranges[{col}] not checkable on gold {n}: {e!r}"[:200])
    ref = os.path.join(spec, "reference.py")
    std = os.path.join(groot, "stdout.txt")
    base = evaluate(t, pred, snap, code_path=ref, stdout_path=std)
    if not correct(base):
        errs.append("gold is not judged correct with the uniform rules: "
                    + str([x["name"] for x in base if x["required"] and not x["passed"]]))
        return errs, notes
    have = {x["name"] for x in base}
    for d in dels:
        for nm in d["checked_by"]:
            if nm not in have:
                errs.append(f"{d['id']}: checked_by {nm!r} is not a check name returned by checks.py")
    # ---- ablation
    for d in dels:
        drop = files_of(names, d["patterns"])
        if not drop:
            continue
        v = view_of(pred, drop=drop)
        try:
            ok = correct(evaluate(t, v, snap, code_path=ref, stdout_path=std))
        finally:
            shutil.rmtree(v, ignore_errors=True)
        if d["required"] and ok:
            errs.append(f"{d['id']}: required, but gold minus {sorted(drop)} is still judged correct")
        if not d["required"] and not ok:
            errs.append(f"{d['id']}: optional, but gold minus {sorted(drop)} is judged wrong (is it required?)")
        notes.append(f"ablate {d['id']}: correct={ok}")
    geo = sorted({n for n, _ in checklib.contract_geo_files(pred, c)
                  if checklib.geo_file_crs(os.path.join(pred, n))[0]})
    for n in geo:
        v = view_of(pred, nocrs=[n])
        try:
            res = evaluate(t, v, snap, code_path=ref, stdout_path=std)
        finally:
            shutil.rmtree(v, ignore_errors=True)
        crs_row = [x for x in res if x["name"] == checklib.CRS_CHECK_NAME]
        if correct(res) or not crs_row or crs_row[0]["passed"]:
            errs.append(f"gold with {n} rewritten without CRS is not rejected by the uniform CRS check")
        notes.append(f"nocrs {n}: correct={correct(res)}")
    return errs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="1-50")
    ap.add_argument("--json", help="write the report here")
    a = ap.parse_args()
    rep, bad = {}, 0
    for t in parse_tasks(a.tasks):
        try:
            errs, notes = validate(t)
        except Exception:
            import traceback
            errs, notes = ["validator crashed: " + traceback.format_exc()[-600:]], []
        rep[t] = {"ok": not errs, "errors": errs, "notes": notes}
        bad += bool(errs)
        print(f"T{t:02d} {'OK ' if not errs else 'BAD'} {'; '.join(errs)[:900]}", flush=True)
    print(f"contracts: {len(rep) - bad}/{len(rep)} consistent")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
