#!/usr/bin/env python3
"""Run a task's control solutions (benchmark/specs/<id>/controls/{correct_*,wrong_*}.py) in
the container with capture and score them with the task's checks. correct_* must be judged
correct, wrong_* must be judged not correct.

    python3 harness/run_controls.py --task 1

The verdicts are written to benchmark/specs/<id>/controls_result.json (or --result). The
working files of a control are removed unless its verdict is not the expected one.
"""
import argparse
import glob
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from capture import run_with_capture  # noqa: E402
from build_gold import evaluate  # noqa: E402
import checklib  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run and score a task's control solutions.")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--controls-dir", help="default benchmark/specs/<id>/controls")
    ap.add_argument("--result", help="default benchmark/specs/<id>/controls_result.json")
    ap.add_argument("--work-dir", help="where the controls run (default benchmark/gold/_controls)")
    a = ap.parse_args()
    t = a.task
    cdir = a.controls_dir or os.path.join(paths.SPECS, str(t), "controls")
    files = sorted(glob.glob(os.path.join(cdir, "*.py")))
    data = os.path.join(paths.INPUTS, str(t), "dataset")
    work_root = a.work_dir or os.path.join(paths.GOLD, "_controls")
    report, ok_all = [], True
    for f in files:
        name = os.path.basename(f)[:-3]
        expect = name.startswith("correct")
        work = os.path.join(work_root, str(t), name)
        run = run_with_capture(open(f).read(), data, work, timeout=1800, cell_id=f"ctl-T{t:02d}")
        res = evaluate(t, os.path.join(work, "pred_results"), os.path.join(work, "_task_root", "_snap"), code_path=f,
                       stdout_path=os.path.join(work, "stdout.txt"))
        v = checklib.verdict(res)          # None = checks.py crashed: never as expected
        got = bool(v)
        good = v is not None and got == expect
        ok_all &= good
        fails = [c["name"] for c in res if c["required"] and not c["passed"]]
        report.append({"control": name, "expected_correct": expect, "judged_correct": got, "as_expected": good,
                       "ran": run["success"], "failed_checks": fails, "scorer_error": v is None})
        if good:  # the snapshots are large: keep the working files only for unexpected verdicts
            shutil.rmtree(work, ignore_errors=True)
        print(f"  [{'OK ' if good else 'BAD'}] {name:30s} expected={'correct' if expect else 'wrong'} "
              f"judged={'SCORER CRASH' if v is None else 'correct' if got else 'wrong'} ran={run['success']} fails={fails}")
    json.dump(report, open(a.result or os.path.join(paths.SPECS, str(t), "controls_result.json"), "w"), indent=1)
    print(f"T{t:02d} controls: {'PASS' if ok_all and files else 'FAIL'} ({len(files)} controls)")
    return 0 if ok_all and files else 1


if __name__ == "__main__":
    sys.exit(main())
