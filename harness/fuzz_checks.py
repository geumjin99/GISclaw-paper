#!/usr/bin/env python3
"""Fuzz the task checks with degenerate agent outputs built from each task's own gold run.

    python3 -W ignore harness/fuzz_checks.py [--tasks 1-50] [--jobs 4]
        [--variants empty,nan_all,...] [--json out.json] [--container]

For every task and variant a work directory is built with the layout of a control run
(run_controls.py): <work>/pred_results/, <work>/_task_root/_snap/, <work>/stdout.txt, and it
is scored with the same call run_controls.py makes - build_gold.evaluate(task, pred_results,
_snap, code_path=..., stdout_path=...). The code text is the task's reference.py (code that
looks right must not rescue degenerate outputs); stdout is empty.

Variants (built from benchmark/gold/<id>/{pred_results, _task_root/_snap}):
  sanity: must be judged CORRECT
    gold                  the gold outputs and snapshot unchanged, gold stdout (checks the
                          fuzz machinery itself)
  must be judged WRONG
    empty                 no output files, no snapshot
    empty_files           every gold output file present but 0 bytes, no snapshot
    empty_files_goldsnap  0-byte output files, gold snapshot (the in-memory work looked right);
                          reported only for tasks whose contract has no required deliverable file
                          (the answer may legitimately live in memory / stdout, e.g. T28)
    nan_all               every output degraded: vectors with missing/empty geometries and NaN
                          attributes, CSV/xlsx numbers NaN, rasters all nodata, figures blank
                          white, other files empty; snapshot arrays/tables/scalars NaN
  must only not crash (the verdict is reported)
    nan_few               1 % of rows (at least one) missing: NaN rows in captured float arrays
                          (e.g. unreachable network nodes drawn at NaN), missing
                          geometries, NaN table rows; rasters / figures unchanged. Ideally
                          still judged as the gold run (reported as a verdict change)
    nan_partial           20 % of rows / pixels / geometries NaN or missing, files and snapshot
    nan_geom              every geometry missing or empty (files and snapshot), rest gold
    nan_arrays            every captured float array NaN (snapshot incl. figure arrays), rest gold
    missing_columns       attribute columns dropped from every vector / table (files and
                          snapshot), rest gold
  --container adds `noop`: a control script that does nothing, run in the container exactly
  like run_controls.py (must be judged wrong).

A scorer crash is a result named "checks.py crashed" (build_gold.evaluate). Exit code 1 when
any variant crashes the scorer or a must-be-wrong variant is judged correct.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
import checklib  # noqa: E402

MUST_BE_WRONG = ("empty", "empty_files", "empty_files_goldsnap", "nan_all", "noop")
VARIANTS = ("gold", "empty", "empty_files", "empty_files_goldsnap", "nan_all",
            "nan_few", "nan_partial", "nan_geom", "nan_arrays", "missing_columns")
BIG = 3e8          # bytes: larger captured arrays are kept as they are (disk / memory)
FRAC = 0.2         # share of rows / pixels degraded by nan_partial
VEC_EXT = (".shp", ".geojson", ".gpkg")
SHP_SIDE = (".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".qmd", ".shp.xml")


def parse_tasks(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


# ----------------------------------------------------------------------------- degraders
def _rng(seed=0):
    return np.random.default_rng(seed)


def _mask(n, mode, rng):
    """Rows to degrade: all (mode 'all') or ~FRAC of them, at least one."""
    if mode == "all" or n == 0:
        return np.ones(n, bool)
    if mode == "few":
        m = np.zeros(n, bool)
        m[rng.choice(n, max(1, n // 100), replace=False)] = True
        return m
    m = rng.random(n) < FRAC
    if not m.any():
        m[0] = True
    return m


def _geom_cols(df):
    try:
        return [df.geometry.name]
    except Exception:
        return []


def degrade_table(df, mode, rng, geom=True, values=True, drop_cols=False):
    """Copy of a (Geo)DataFrame / Series with numeric values NaN and/or geometries missing /
    empty on the rows chosen by mode; drop_cols keeps only the geometry (or one index column)."""
    import pandas as pd
    if isinstance(df, pd.Series):
        df = df.to_frame()
    df = df.copy()
    gcols = _geom_cols(df)
    if drop_cols:
        if gcols:
            return df[gcols]
        return pd.DataFrame({"id": np.arange(len(df))})
    m = _mask(len(df), mode, rng)
    if values:
        for c in df.columns:
            if c in gcols:
                continue
            if df[c].dtype.kind in "fiub":
                col = df[c].astype(float)
                col[m] = np.nan
                df[c] = col
    if geom and gcols:
        from shapely.geometry import Point
        gs = df.geometry.copy()
        idx = np.flatnonzero(m)
        gs.iloc[idx[0::2]] = None            # missing
        gs.iloc[idx[1::2]] = Point()         # empty
        df = df.set_geometry(gs)
    return df


def degrade_array(a, mode, rng):
    a = np.array(a, copy=True)
    if a.dtype.kind == "f" and mode == "few":
        if a.ndim >= 1 and len(a) >= 10:
            a[_mask(len(a), "few", rng)] = np.nan
    elif a.dtype.kind == "f":
        if mode == "all":
            a[...] = np.nan
        else:
            a[rng.random(a.shape) < FRAC] = np.nan
    elif a.dtype.kind in "iub" and mode == "all":
        a[...] = 0
    return a


def write_vector(gdf, path):
    ext = os.path.splitext(path)[1].lower()
    for p in [path] + ([path[:-4] + s for s in SHP_SIDE] if ext == ".shp" else []):
        if os.path.lexists(p):
            os.remove(p)
    try:
        gdf.to_file(path)
    except Exception:
        # some drivers refuse empty geometries: use missing ones instead
        g = gdf.copy()
        g.geometry.iloc[np.flatnonzero(g.geometry.is_empty.fillna(False).to_numpy())] = None
        g.to_file(path)


def _is_geojson(path):
    try:
        with open(path, errors="ignore") as f:
            head = f.read(4096)
        return '"type"' in head and ("Feature" in head or '"coordinates"' in head)
    except OSError:
        return False


def degrade_file(src, dst, variant, rng):
    """Write the degraded version of gold file src to dst (vector files: all sidecars)."""
    name = os.path.basename(src)
    low = name.lower()
    ext = os.path.splitext(low)[1]
    if variant in ("empty_files", "empty_files_goldsnap"):
        open(dst, "wb").close()
        return
    is_vec = ext in VEC_EXT or (ext == ".json" and _is_geojson(src))
    if low.endswith(SHP_SIDE):
        return                                # written with their .shp
    if variant in ("nan_all", "nan_partial", "nan_few", "nan_geom", "missing_columns") and is_vec:
        import geopandas as gpd
        g = gpd.read_file(src)
        mode = {"nan_partial": "partial", "nan_few": "few"}.get(variant, "all")
        g = degrade_table(g, mode, rng, geom=variant != "missing_columns",
                          values=variant in ("nan_all", "nan_partial", "nan_few"), drop_cols=variant == "missing_columns")
        write_vector(g, dst)
        return
    if variant in ("nan_all", "nan_partial", "nan_few", "missing_columns") and ext in (".csv", ".xlsx", ".xls"):
        import pandas as pd
        df = pd.read_csv(src, sep=None, engine="python") if ext == ".csv" else pd.read_excel(src)
        df = degrade_table(df, {"nan_partial": "partial", "nan_few": "few"}.get(variant, "all"), rng, geom=False,
                           drop_cols=variant == "missing_columns")
        if ext == ".csv":
            df.to_csv(dst, index=False)
        else:
            df.to_excel(dst if ext == ".xlsx" else dst[:-4] + ".xlsx", index=False)
        return
    if variant in ("nan_all", "nan_partial") and ext in (".tif", ".tiff"):
        import rasterio
        with rasterio.open(src) as r:
            prof = r.profile.copy()
            a = r.read()
        if a.dtype.kind == "f":
            nod = np.nan
        else:
            nod = prof.get("nodata")
            if nod is None:
                nod = 0
        prof["nodata"] = nod
        m = np.ones(a.shape, bool) if variant == "nan_all" else rng.random(a.shape) < FRAC
        a[m] = nod
        with rasterio.open(dst, "w", **prof) as w:
            w.write(a)
        return
    if variant == "nan_all":
        if ext in (".png", ".jpg", ".jpeg"):
            from PIL import Image
            with Image.open(src) as im:
                Image.new("RGB", im.size, "white").save(dst)
            return
        if ext == ".json":
            def strip(x):
                if isinstance(x, dict):
                    return {k: strip(v) for k, v in x.items()}
                if isinstance(x, list):
                    return [strip(v) for v in x]
                return None if isinstance(x, (int, float)) and not isinstance(x, bool) else x
            try:
                json.dump(strip(json.load(open(src))), open(dst, "w"))
                return
            except (ValueError, OSError):
                pass
        open(dst, "wb").close()               # text, html, pdf, ...: empty
        return
    os.symlink(src, dst)                      # unchanged


def build_pred(gold_pred, dst, variant, rng):
    if variant == "gold":
        os.symlink(gold_pred, dst)
        return
    os.makedirs(dst)
    if variant == "empty" or not os.path.isdir(gold_pred):
        return
    for n in sorted(os.listdir(gold_pred)):
        src = os.path.join(gold_pred, n)
        if os.path.isdir(src):
            if variant in ("nan_all", "empty_files", "empty_files_goldsnap"):
                continue                      # sub-folders: dropped by the degenerate agent
            os.symlink(src, os.path.join(dst, n))
            continue
        degrade_file(src, os.path.join(dst, n), variant, rng)


def build_snap(gold_snap, dst, variant, rng):
    """Snapshot root for the variant, or None (no snapshot)."""
    if variant in ("empty", "empty_files"):
        return None
    if variant in ("gold", "empty_files_goldsnap") or not os.path.isdir(gold_snap):
        return gold_snap
    os.makedirs(dst)
    mode = {"nan_partial": "partial", "nan_few": "few"}.get(variant, "all")
    for tag in sorted(os.listdir(gold_snap)):
        s = os.path.join(gold_snap, tag)
        d = os.path.join(dst, tag)
        idx_p = os.path.join(s, "index.json")
        if not os.path.isdir(s) or not os.path.exists(idx_p):
            os.symlink(s, d)                  # savefig.json and friends
            continue
        os.makedirs(d)
        index = json.load(open(idx_p))
        done = set()
        for m in index:
            k = m.get("kind")
            fn = m["name"] + (".npy" if k == "array" else ".parquet")
            src, out = os.path.join(s, fn), os.path.join(d, fn)
            if fn in done or not os.path.exists(src):
                continue
            done.add(fn)
            try:
                if k == "array" and variant in ("nan_all", "nan_partial", "nan_few", "nan_arrays") \
                        and os.path.getsize(src) < BIG:
                    np.save(out, degrade_array(np.load(src, allow_pickle=False), mode, rng))
                elif k == "gdf" and variant in ("nan_all", "nan_partial", "nan_few", "nan_geom", "missing_columns"):
                    import geopandas as gpd
                    g = degrade_table(gpd.read_parquet(src), mode, rng, geom=variant != "missing_columns",
                                      values=variant in ("nan_all", "nan_partial", "nan_few"),
                                      drop_cols=variant == "missing_columns")
                    g.to_parquet(out)
                elif k in ("df", "series") and variant in ("nan_all", "nan_partial", "nan_few", "missing_columns"):
                    import pandas as pd
                    t = degrade_table(pd.read_parquet(src), mode, rng, geom=False,
                                      drop_cols=variant == "missing_columns")
                    t.columns = [str(c) for c in t.columns]
                    t.to_parquet(out)
                else:
                    os.symlink(src, out)
            except Exception:
                if not os.path.lexists(out):
                    os.symlink(src, out)      # could not degrade (e.g. object array): keep
        new_index = []
        for m in index:
            m = dict(m)
            if variant in ("nan_all", "nan_partial") and m.get("kind") == "scalar" \
                    and isinstance(m.get("value"), (int, float)) and not isinstance(m.get("value"), bool):
                if variant == "nan_all" or rng.random() < FRAC:
                    m["value"] = float("nan")
            if variant == "missing_columns" and m.get("kind") in ("gdf", "df", "series"):
                m["columns"] = m.get("columns", [])[-1:] if m.get("kind") == "gdf" else ["id"]
            new_index.append(m)
        json.dump(new_index, open(os.path.join(d, "index.json"), "w"))
        for n in os.listdir(s):               # anything else in the tag folder
            if not os.path.lexists(os.path.join(d, n)):
                os.symlink(os.path.join(s, n), os.path.join(d, n))
    return dst


# ----------------------------------------------------------------------------- one task
def has_required_file(t):
    try:
        c = json.load(open(os.path.join(paths.SPECS, str(t), "contract.json")))
        return any(d.get("required") for d in c.get("deliverables", []))
    except (OSError, ValueError):
        return True


def fuzz_task(t, variants, scratch, container=False):
    from build_gold import evaluate
    gold = os.path.join(paths.GOLD, str(t))
    ref = os.path.join(paths.SPECS, str(t), "reference.py")
    rows = []
    for v in variants:
        work = os.path.join(scratch, f"T{t:02d}", v)
        shutil.rmtree(work, ignore_errors=True)
        rec = {"task": t, "variant": v, "must_be_wrong": v in MUST_BE_WRONG}
        if v == "empty_files_goldsnap" and not has_required_file(t):
            rec["must_be_wrong"] = False
        t0 = time.time()
        try:
            if v == "noop":
                from capture import run_with_capture
                code = os.path.join(work + "_code.py")
                os.makedirs(os.path.dirname(code), exist_ok=True)
                open(code, "w").write("print('nothing done')\n")
                data = os.path.join(paths.INPUTS, str(t), "dataset")
                run_with_capture(open(code).read(), data, work, timeout=600, cell_id=f"fuzz-T{t:02d}")
                pred, snap, std = os.path.join(work, "pred_results"), os.path.join(work, "_task_root", "_snap"), \
                    os.path.join(work, "stdout.txt")
                cpath = code
            else:
                rng = _rng(t)
                pred = os.path.join(work, "pred_results")
                os.makedirs(work)
                build_pred(os.path.join(gold, "pred_results"), pred, v, rng)
                snap = build_snap(os.path.join(gold, "_task_root", "_snap"),
                                  os.path.join(work, "_task_root", "_snap"), v, rng)
                std = os.path.join(work, "stdout.txt")
                if v == "gold":
                    shutil.copy(os.path.join(gold, "stdout.txt"), std)
                else:
                    open(std, "w").close()
                cpath = ref
        except Exception as e:
            import traceback
            rec.update(build_error=repr(e)[:300], tb=traceback.format_exc()[-600:])
            rows.append(rec)
            shutil.rmtree(work, ignore_errors=True)
            continue
        # the same call as run_controls.py
        res = evaluate(t, pred, snap, code_path=cpath, stdout_path=std)
        req = [c for c in res if c["required"]]
        rec["crashed"] = checklib.has_scorer_error(res)
        rec["judged_correct"] = (not rec["crashed"]) and bool(req) and all(c["passed"] for c in req)
        rec["failed"] = [c["name"] for c in req if not c["passed"]]
        if rec["crashed"]:
            rec["crash_note"] = next(c["note"] for c in res if c.get("scorer_error"))[-600:]
        rec["seconds"] = round(time.time() - t0, 1)
        rows.append(rec)
        shutil.rmtree(work, ignore_errors=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="1-50")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=3600, help="per task, seconds")
    ap.add_argument("--container", action="store_true", help="also run the `noop` control in the container")
    ap.add_argument("--json", help="write all rows here")
    ap.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--scratch", help=argparse.SUPPRESS)
    a = ap.parse_args()
    variants = [v for v in a.variants.split(",") if v]
    if a.container and "noop" not in variants:
        variants.append("noop")
    if a.worker is not None:     # one task, rows as JSON on the last stdout line
        rows = fuzz_task(a.worker, variants, a.scratch, a.container)
        print("\n@@ROWS@@" + json.dumps(rows))
        return 0
    scratch = tempfile.mkdtemp(prefix="fuzz_")
    tasks = parse_tasks(a.tasks)

    def one(t):
        cmd = [sys.executable, "-W", "ignore", os.path.abspath(__file__), "--worker", str(t), "--scratch", scratch,
               "--variants", ",".join(variants)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            line = [x for x in p.stdout.splitlines() if x.startswith("@@ROWS@@")]
            if line:
                return json.loads(line[-1][len("@@ROWS@@"):])
            return [{"task": t, "variant": "*", "build_error": "worker failed: " + p.stderr[-600:]}]
        except subprocess.TimeoutExpired:
            return [{"task": t, "variant": "*", "build_error": f"timeout {a.timeout}s"}]
    from concurrent.futures import ThreadPoolExecutor
    allrows, bad = [], 0
    jobs = 1 if a.container else a.jobs   # one container task at a time
    with ThreadPoolExecutor(jobs) as ex:
        for rows in ex.map(one, tasks):
            for r in rows:
                allrows.append(r)
                flag = ""
                if r.get("build_error"):
                    flag = "BUILD-ERROR " + r["build_error"][:200]
                elif r["crashed"]:
                    flag = "CRASH " + r["crash_note"].strip().splitlines()[-1][:200]
                    bad += 1
                elif r["variant"] == "gold" and not r["judged_correct"]:
                    flag = "GOLD-JUDGED-WRONG " + str(r["failed"])[:200]
                    bad += 1
                elif r["must_be_wrong"] and r["judged_correct"]:
                    flag = "JUDGED-CORRECT"
                    bad += 1
                elif r["variant"] == "nan_few" and not r["judged_correct"]:
                    flag = "nan_few: wrong (gold is correct) " + str(r["failed"])[:160]
                elif r.get("judged_correct") and r["variant"] not in ("gold", "nan_few"):
                    flag = "correct (to inspect)"
                print(f"T{r['task']:02d} {r['variant']:22s} "
                      + ("" if r.get("build_error") else f"correct={r['judged_correct']!s:5s} crashed={r['crashed']!s:5s} "
                         f"{r['seconds']:6.1f}s ") + flag, flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    n_cr = sum(1 for r in allrows if r.get("crashed"))
    n_wc = sum(1 for r in allrows if r.get("must_be_wrong") and r.get("judged_correct"))
    n_be = sum(1 for r in allrows if r.get("build_error"))
    n_gw = sum(1 for r in allrows if r.get("variant") == "gold" and r.get("judged_correct") is False)
    print(f"fuzz: {len(allrows)} runs, scorer crashes {n_cr}, must-be-wrong judged correct {n_wc}, "
          f"gold judged wrong {n_gw}, build errors {n_be}")
    if a.json:
        json.dump(allrows, open(a.json, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
