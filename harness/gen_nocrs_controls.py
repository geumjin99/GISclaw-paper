#!/usr/bin/env python3
"""Write the negative control "deliverable written without CRS" for every task whose contract
declares a geographic deliverable:
benchmark/specs/<id>/controls/wrong_deliverable_no_crs.py = the task's reference.py, then
every geographic deliverable file it wrote is rewritten without CRS
(nocrs.write_without_crs). Expected verdict: wrong (the uniform CRS check fails).

    python3 harness/gen_nocrs_controls.py --tasks 1-50
"""
import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
import checklib  # noqa: E402
import nocrs  # noqa: E402

NAME = "wrong_deliverable_no_crs.py"
TAIL = '''

# ---------------------------------------------------------------------------------------------
# NEGATIVE CONTROL (written by harness/gen_nocrs_controls.py from reference.py and
# contract.json; do not edit by hand): the analysis above is the reference, but every
# geographic deliverable file is then written without a CRS. Expected: wrong.
{func}

def _strip_deliverable_crs(patterns):
    import fnmatch
    import os
    for n in sorted(os.listdir("pred_results")):
        low = n.lower()
        if low.endswith({exts!r}) and any(fnmatch.fnmatchcase(low, p.lower()) for p in patterns):
            if low.endswith(".json"):
                try:
                    import json
                    if json.load(open(os.path.join("pred_results", n))).get("type") is None:
                        continue
                except Exception:
                    continue
            print("no-CRS control:", n, _write_without_crs(os.path.join("pred_results", n)))


_strip_deliverable_crs({patterns!r})
'''


def parse_tasks(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description="Write the no-CRS negative control of each task.")
    ap.add_argument("--tasks", default="1-50")
    a = ap.parse_args()
    func = inspect.getsource(nocrs.write_without_crs).replace("def write_without_crs(", "def _write_without_crs(", 1)
    for t in parse_tasks(a.tasks):
        spec = os.path.join(paths.SPECS, str(t))
        c = checklib.load_contract(t)
        out = os.path.join(spec, "controls", NAME)
        pats = sorted({p for d in (c or {}).get("deliverables", []) if d.get("crs_required") for p in d["patterns"]})
        if not pats:
            if os.path.exists(out):
                os.remove(out)
            print(f"T{t:02d} no geographic deliverable - no control")
            continue
        gold = os.path.join(paths.GOLD, str(t), "pred_results")
        hit = [n for n, _ in checklib.contract_geo_files(gold, c) if checklib.geo_file_crs(os.path.join(gold, n))[0]]
        if not hit:
            print(f"T{t:02d} WARNING: the reference writes no geographic deliverable file - no control")
            continue
        ref = open(os.path.join(spec, "reference.py")).read().rstrip() + "\n"
        open(out, "w").write(ref + TAIL.format(func=func, patterns=pats,
                                               exts=checklib.VECTOR_EXT + checklib.RASTER_EXT))
        print(f"T{t:02d} {NAME}: strips {sorted(set(hit))}")


if __name__ == "__main__":
    main()
