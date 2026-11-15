#!/usr/bin/env python3
"""Build the task variants of the perturbation test.

    python3 harness/make_perturbations.py                # task table, inputs and specs
    python3 harness/make_perturbations.py --skip-inputs  # task table and specs only

Ten base tasks are varied in up to three ways. A variant has the id 100*k + t, where t is
the base task and k the kind of perturbation:

  1  reprojected inputs    every vector layer the reference solution reads is written in
                           another coordinate reference system
  2  renamed fields        the attribute fields the analysis uses are renamed in the data
                           and in the task text
  3  reworded instruction  the instruction is rephrased; data and checks are unchanged

What is varied is listed in benchmark/perturbations/variants.json (reprojection targets and
field renames) and benchmark/perturbations/reworded_instructions.json. For every variant N
this script writes

  benchmark/perturbations/tasks.csv   the task rows of all variants (id = N)
  benchmark/inputs/<N>/dataset        the varied copy of the base inputs (read-only)
  benchmark/specs/<N>/                the base spec (reference solution, checks, contract,
                                      control solutions) adapted as described below

so that a variant is run, given a gold and scored like any other task.

How the specs are adapted, so that every check keeps its meaning
  reprojected inputs
    inputs     vector layers are reprojected with ogr2ogr. The attribute table of a
               shapefile is kept byte for byte, and the sidecar files that describe the
               old coordinates (.sbn/.sbx spatial index, .shp.xml metadata) are removed.
               Rasters, tables and text files are unchanged.
    reference  where it relied on the coordinate system of the base inputs it reprojects
               explicitly (REFERENCE_EDITS).
    checks     unchanged, but they read the inputs of the base task: their reference
               values are defined in the base coordinate systems, and the data are the same.
    controls   unchanged code behind a preamble that reprojects each layer back to its
               base coordinate system when it is read, that is, the same solution with a
               correct reprojection step.
  renamed fields
    inputs     the fields are renamed in the data (GeoJSON keys, DBF header, GeoPackage
               column, CSV header, xlsx header cells, .shp.xml metadata).
    task text  the same names are substituted in Instruction and Dataset Description.
    reference, checks, controls, spec.json
               the same names are substituted (quoted literals and attribute access;
               distinctive names also as bare words).
  reworded instruction
    only the Instruction differs from the base task.

Requires the base inputs (harness/prepare_inputs.py) and the GDAL command-line tool
ogr2ogr. The output is deterministic for a given GDAL version.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

VARIANTS = os.path.join(paths.PERTURBATIONS, "variants.json")
REWORDED = os.path.join(paths.PERTURBATIONS, "reworded_instructions.json")
VARIANT_TASKS_CSV = os.path.join(paths.PERTURBATIONS, "tasks.csv")

REPROJECTED, RENAMED, REWORDED_KIND = "reprojected inputs", "renamed fields", "reworded instruction"

# Lines of a reference solution that assumed the coordinate system of the base inputs, and
# their replacement in the reprojected-inputs variant, by base task.
REFERENCE_EDITS = {
    19: [('samples = gpd.read_file("dataset/DissolvedO2.geojson")',
          'samples = gpd.read_file("dataset/DissolvedO2.geojson").to_crs("EPSG:32618")')],
    20: [('occ = gpd.read_file("dataset/sn_w_minoccs.gpkg")',
          'occ = gpd.read_file("dataset/sn_w_minoccs.gpkg").to_crs("EPSG:28355")')],
    29: [('paths = gpd.read_file("dataset/TsuniamiPaths.gpkg").set_crs(crs, allow_override=True)',
          'paths = gpd.read_file("dataset/TsuniamiPaths.gpkg").to_crs(crs)'),
         ('sources = gpd.read_file("dataset/TsunamiSources.gpkg").set_crs(crs, allow_override=True)',
          'sources = gpd.read_file("dataset/TsunamiSources.gpkg").to_crs(crs)')],
    42: [("nbhd = gpd.read_file('dataset/berlin-neighbourhoods.geojson')",
          "nbhd = gpd.read_file('dataset/berlin-neighbourhoods.geojson').to_crs('EPSG:4326')")],
}

# Field names that are also common words: in code they are substituted only as quoted
# literals and attribute access, never as bare words.
COMMON_WORDS = {"Species", "Station", "Route", "Name", "State", "Change", "price", "latitude", "longitude",
                "neighbourhood"}

DRIVERS = {".geojson": "GeoJSON", ".shp": "ESRI Shapefile", ".gpkg": "GPKG"}


def read_text(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def write_text(p, text):
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)


def run(*args):
    # A fixed date for the modification time that GDAL stores in a GeoPackage, so that the
    # files are the same on every run.
    env = dict(os.environ, OGR_CURRENT_DATE="1970-01-01T00:00:00.000Z")
    subprocess.run(args, check=True, capture_output=True, env=env)


def set_writable(d, writable):
    for root, _, files in os.walk(d):
        for f in files:
            os.chmod(os.path.join(root, f), 0o644 if writable else 0o444)
        os.chmod(root, 0o755 if writable else 0o555)


# ----------------------------------------------------------------------------- inputs
def copy_inputs(base, variant):
    """Writable copy of the base inputs as the inputs of the variant."""
    src = os.path.join(paths.INPUTS, str(base), "dataset")
    if not os.path.isdir(src):
        raise SystemExit(f"missing base inputs: {src}")
    top = os.path.join(paths.INPUTS, str(variant))
    if os.path.exists(top):
        set_writable(top, True)
        shutil.rmtree(top)
    dst = os.path.join(top, "dataset")
    shutil.copytree(src, dst)
    set_writable(top, True)
    return dst


def reproject(d, fname, crs):
    p = os.path.join(d, fname)
    stem, ext = os.path.splitext(p)
    tmp = os.path.join(d, "__tmp__")
    os.makedirs(tmp)
    out = os.path.join(tmp, fname)
    args = ["ogr2ogr", "-f", DRIVERS[ext], "-t_srs", crs, out, p]
    if ext == ".gpkg":
        args.append("-preserve_fid")
    run(*args)
    if ext == ".shp":
        # Only the geometry changes: the attribute table (.dbf/.cpg) of the base layer is
        # kept, because ogr2ogr narrows float widths and adds an FID field to layers that
        # have no attribute table.
        new_stem = os.path.join(tmp, os.path.splitext(fname)[0])
        for e in (".dbf", ".cpg"):
            if os.path.exists(new_stem + e):
                os.remove(new_stem + e)
        for e in (".shp", ".shx", ".prj", ".sbn", ".sbx", ".shp.xml"):
            if os.path.exists(stem + e):
                os.remove(stem + e)
        for f in os.listdir(tmp):
            shutil.move(os.path.join(tmp, f), os.path.join(d, f))
    else:
        os.remove(p)
        shutil.move(out, p)
    shutil.rmtree(tmp)


def rename_geojson(p, ren):
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    for feat in d["features"]:
        feat["properties"] = {ren.get(k, k): v for k, v in feat["properties"].items()}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)


def rename_dbf(p, ren):
    with open(p, "rb") as f:
        b = bytearray(f.read())
    header = int.from_bytes(b[8:10], "little")
    done = set()
    for off in range(32, header - 1, 32):   # one 32-byte descriptor per field
        name = bytes(b[off:off + 11]).split(b"\0")[0].decode("ascii")
        if name in ren:
            new = ren[name].encode("ascii")
            assert len(new) <= 10, new
            b[off:off + 11] = new + b"\0" * (11 - len(new))
            done.add(name)
    assert done == set(ren), (p, done)
    with open(p, "wb") as f:
        f.write(bytes(b))


def rename_gpkg(p, ren):
    con = sqlite3.connect(p)
    (table,) = con.execute("select table_name from gpkg_contents where data_type='features'").fetchone()
    for old, new in ren.items():
        con.execute(f'ALTER TABLE "{table}" RENAME COLUMN "{old}" TO "{new}"')
    if con.execute("select count(*) from sqlite_master where name='gpkg_data_columns'").fetchone()[0]:
        for old, new in ren.items():
            con.execute("update gpkg_data_columns set column_name=? where table_name=? and column_name=?",
                        (new, table, old))
    con.commit()
    con.close()


def rename_csv_header(p, ren):
    with open(p, "rb") as f:
        raw = f.read()
    i = raw.index(b"\n")
    cols = raw[:i].decode("utf-8").split(",")
    new = []
    for c in cols:
        quoted = c.startswith('"') and c.endswith('"')
        name = c.strip('"').strip("\r")
        r = ren.get(name, name)
        new.append(f'"{r}"' if quoted else r + ("\r" if c.endswith("\r") else ""))
    assert sum(1 for c in cols if c.strip('"').strip("\r") in ren) == len(ren), p
    with open(p, "wb") as f:
        f.write(",".join(new).encode("utf-8") + raw[i:])


def rename_xlsx_header(p, ren):
    tmp = p + ".tmp"
    with zipfile.ZipFile(p) as zi, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zo:
        for item in zi.infolist():
            data = zi.read(item.filename)
            if item.filename == "xl/sharedStrings.xml":
                s = data.decode("utf-8")
                for old, new in ren.items():
                    assert s.count(f"<t>{old}</t>") == 1, old
                    s = s.replace(f"<t>{old}</t>", f"<t>{new}</t>")
                data = s.encode("utf-8")
            zo.writestr(item, data)
    os.replace(tmp, p)


def rename_fields(d, fname, ren):
    p = os.path.join(d, fname)
    if fname.endswith(".geojson"):
        rename_geojson(p, ren)
    elif fname.endswith(".shp"):
        rename_dbf(p[:-4] + ".dbf", ren)
        if os.path.exists(p + ".xml"):
            write_text(p + ".xml", subst_text(read_text(p + ".xml"), ren))
    elif fname.endswith(".gpkg"):
        rename_gpkg(p, ren)
    elif fname.endswith(".csv"):
        rename_csv_header(p, ren)
    elif fname.endswith(".xlsx"):
        rename_xlsx_header(p, ren)
    else:
        raise SystemExit(f"no field renaming for {fname}")


# ----------------------------------------------------------------------------- text and code
def subst_code(text, ren):
    """Field names in code: quoted literals and attribute access; distinctive names also bare."""
    for old in sorted(ren, key=len, reverse=True):
        new = ren[old]
        text = re.sub(r"(['\"])" + re.escape(old) + r"\1", lambda m: m.group(1) + new + m.group(1), text)
        text = re.sub(r"\." + re.escape(old) + r"\b", "." + new, text)
        if old not in COMMON_WORDS:
            text = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])", new, text)
    return text


def subst_text(text, ren):
    for old in sorted(ren, key=len, reverse=True):
        text = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(old) + r"(?![A-Za-z0-9_])", ren[old], text)
    return text


# ----------------------------------------------------------------------------- specs
def copy_spec(base, variant):
    src, dst = os.path.join(paths.SPECS, str(base)), os.path.join(paths.SPECS, str(variant))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "controls_result.json"))
    p = os.path.join(dst, "contract.json")
    with open(p, encoding="utf-8") as f:
        c = json.load(f)
    c["task"] = variant
    c["base_task"] = base
    with open(p, "w", encoding="utf-8") as f:
        json.dump(c, f, indent=1)
    return dst


def spec_files(spec_dir):
    return ["reference.py", "checks.py", "spec.json"] + \
        ["controls/" + c for c in sorted(os.listdir(os.path.join(spec_dir, "controls")))]


def control_preamble(original_crs):
    return ("# Reprojected-inputs variant: each layer is reprojected back to the coordinate reference\n"
            "# system of the base task's inputs when it is read.\n"
            "import os as _variant_os, geopandas as _variant_gpd\n"
            f"_VARIANT_BASE_CRS = {json.dumps(original_crs)}\n"
            "_variant_read_file = _variant_gpd.read_file\n"
            "def _variant_read_in_base_crs(path, *a, **k):\n"
            "    g = _variant_read_file(path, *a, **k)\n"
            "    c = _VARIANT_BASE_CRS.get(_variant_os.path.basename(str(path)))\n"
            "    return g.to_crs(c) if (c and g.crs is not None) else g\n"
            "_variant_gpd.read_file = _variant_read_in_base_crs\n"
            "# End of the variant preamble.\n")


def checks_on_base_inputs(base):
    return ("\n\n# Reprojected-inputs variant: the checks read the inputs of the base task "
            f"({base}), which are the\n"
            "# same data in the coordinate reference systems the reference values are defined in.\n"
            "import os as _variant_os\n"
            "_variant_base_checks = checks\n\n\n"
            "def checks(ctx):\n"
            "    inputs_root = _variant_os.path.dirname(_variant_os.path.dirname(_variant_os.path.normpath(ctx.inputs_dir)))\n"
            f"    ctx.inputs_dir = _variant_os.path.join(inputs_root, '{base}', 'dataset')\n"
            "    return _variant_base_checks(ctx)\n")


def spec_reprojected(base, variant, layers):
    s = copy_spec(base, variant)
    p = os.path.join(s, "reference.py")
    ref = read_text(p)
    for old, new in REFERENCE_EDITS.get(base, []):
        assert ref.count(old) == 1, (base, old)
        ref = ref.replace(old, new)
    write_text(p, ref)
    p = os.path.join(s, "checks.py")
    write_text(p, read_text(p) + checks_on_base_inputs(base))
    preamble = control_preamble({f: v["from"] for f, v in layers.items()})
    cdir = os.path.join(s, "controls")
    for c in sorted(os.listdir(cdir)):
        p = os.path.join(cdir, c)
        write_text(p, preamble + read_text(p))


def spec_renamed(base, variant, ren):
    s = copy_spec(base, variant)
    for rel in spec_files(s):
        p = os.path.join(s, rel)
        write_text(p, subst_code(read_text(p), ren))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Build the task variants of the perturbation test.")
    ap.add_argument("--skip-inputs", action="store_true",
                    help="write the variant task table and specs only (needs neither the base inputs nor ogr2ogr)")
    a = ap.parse_args()
    if not a.skip_inputs and not shutil.which("ogr2ogr"):
        raise SystemExit("ogr2ogr (GDAL command-line tools) not found")
    with open(VARIANTS, encoding="utf-8") as f:
        variants = json.load(f)
    with open(REWORDED, encoding="utf-8") as f:
        reworded = json.load(f)
    with open(paths.TASKS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = [c for c in reader.fieldnames if c != "CodeString"]
        rows = {int(r["id"]): r for r in reader}

    out_rows = []
    for key, v in variants.items():
        n, t, kind = int(key), v["base"], v["perturbation"]
        if "not_applicable" in v:
            continue
        row = {c: rows[t][c] for c in fields}
        row["id"] = str(n)
        if not a.skip_inputs:
            d = copy_inputs(t, n)
        if kind == REPROJECTED:
            if not a.skip_inputs:
                for fname, crs in v["reprojected"].items():
                    reproject(d, fname, crs["to"])
            spec_reprojected(t, n, v["reprojected"])
        elif kind == RENAMED:
            ren = {}
            for fname, r in v["renamed"].items():
                ren.update(r)
                if not a.skip_inputs:
                    rename_fields(d, fname, r)
            spec_renamed(t, n, ren)
            for c in ("Instruction", "Dataset Description"):
                row[c] = subst_text(row[c], ren)
        elif kind == REWORDED_KIND:
            copy_spec(t, n)
            row["Instruction"] = reworded[str(t)]
        else:
            raise SystemExit(f"variant {n}: unknown perturbation {kind!r}")
        if not a.skip_inputs:
            set_writable(os.path.join(paths.INPUTS, str(n)), False)
        out_rows.append(row)

    with open(VARIANT_TASKS_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(out_rows)} variants: {' '.join(r['id'] for r in out_rows)}")
    print(f"task table: {VARIANT_TASKS_CSV}")
    print(f"specs:      {paths.SPECS}/<id>")
    if not a.skip_inputs:
        print(f"inputs:     {paths.INPUTS}/<id>/dataset")


if __name__ == "__main__":
    main()
