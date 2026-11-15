#!/usr/bin/env python3
"""Build the gold of every task and run its control solutions, a few tasks at a time.

    python3 harness/build_all_gold.py [--tasks 1-50] [--jobs 3] [--json summary.json]

Per task: build_gold.py (reference solution, self-test of the checks), then run_controls.py.
Exit code 1 when any task fails either step.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402


def parse_tasks(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def one(t):
    t0 = time.time()
    rec = {"task": t}
    for step, script in (("gold", "build_gold.py"), ("controls", "run_controls.py")):
        p = subprocess.run([sys.executable, "-W", "ignore", os.path.join(paths.HARNESS_DIR, script), "--task", str(t)],
                           capture_output=True, text=True)
        rec[step] = {"rc": p.returncode, "tail": p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]}
    rec["ok"] = rec["gold"]["rc"] == 0 and rec["controls"]["rc"] == 0
    rec["elapsed_s"] = round(time.time() - t0)
    print(f"T{t:02d} ok={rec['ok']} {rec['gold']['tail']} | {rec['controls']['tail']} ({rec['elapsed_s']}s)", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser(description="Build every task's gold and run its controls.")
    ap.add_argument("--tasks", default="1-50")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--json", help="write the per-task summary here")
    a = ap.parse_args()
    with ThreadPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, parse_tasks(a.tasks)))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    bad = [r["task"] for r in res if not r["ok"]]
    print("ALL OK" if not bad else f"FAILED: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
