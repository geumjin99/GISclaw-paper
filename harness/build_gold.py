#!/usr/bin/env python3
"""Run a task's reference solution in the container with capture, store its outputs as the
task's gold, and self-test the task's checks against it (gold vs gold must pass every
required check).

    python3 harness/build_gold.py --task 1

Inputs : benchmark/inputs/<id>/dataset                 (read-only)
         benchmark/specs/<id>/reference.py             (reference solution)
         benchmark/specs/<id>/checks.py                (correctness checks)
Outputs: benchmark/gold/<id>/{pred_results/, _task_root/_snap/, run.json, selftest.json}
"""
import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from capture import run_with_capture  # noqa: E402
import checklib  # noqa: E402


def load_checks(tid):
    p = os.path.join(paths.SPECS, str(tid), "checks.py")
    spec = importlib.util.spec_from_file_location(f"checks_{tid}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.checks


def evaluate(tid, pred_dir, pred_snap, gold_root=None, code_path=None, stdout_path=None,
             exclude_prefixes=(), uniform=True):
    """Run the task's checks, plus the uniform rules (checklib) unless uniform=False: files
    that copy an input dataset are hidden from the checks, and every geographic deliverable
    file declared in contract.json must carry a CRS.
    exclude_prefixes: also hide files whose name starts with one of these (e.g. 'emergency_',
    to see whether a verdict depends on the harness's emergency save)."""
    import shutil
    gold_root = gold_root or os.path.join(paths.GOLD, str(tid))
    inputs_dir = os.path.join(paths.INPUTS, str(tid), "dataset")
    hide, copies, view = set(), {}, None
    if pred_dir and os.path.isdir(pred_dir):
        if uniform:
            copies = checklib.input_copies(pred_dir, inputs_dir)
            hide |= set(copies)
        if exclude_prefixes:
            hide |= {n for n in os.listdir(pred_dir) if n.startswith(tuple(exclude_prefixes))}
        if hide:
            view = checklib.scoring_view(pred_dir, hide)
    ctx = checklib.Ctx(pred_dir=view or pred_dir, gold_dir=os.path.join(gold_root, "pred_results"),
                       inputs_dir=inputs_dir,
                       pred_snap=pred_snap, gold_snap=os.path.join(gold_root, "_task_root", "_snap"),
                       code_path=code_path, stdout_path=stdout_path)
    try:
        # a crash of the scorer is recorded as such (scorer_error=True), never as a wrong answer
        try:
            res = load_checks(tid)(ctx)
        except Exception:
            import traceback
            return [checklib.scorer_error_result(traceback.format_exc()[-800:])]
        if uniform:
            try:
                c = checklib.deliverable_crs_check(view or pred_dir, checklib.load_contract(tid))
            except Exception:
                import traceback
                return res + [checklib.scorer_error_result("uniform CRS rule: " + traceback.format_exc()[-800:])]
            if c is not None:
                res.append(c)
            if copies:
                r = checklib.result(checklib.COPY_INFO_NAME, True, value=copies, required=False,
                                    note="hidden from the task checks: " + ", ".join(sorted(copies)))
                r["info"] = True
                res.append(r)
        return res
    finally:
        if view:
            shutil.rmtree(view, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Build a task's gold from its reference solution and self-test its checks.")
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--timeout", type=int, default=1800, help="time limit of the reference solution, seconds")
    ap.add_argument("--skip-run", action="store_true", help="only run the self-test on the existing gold")
    a = ap.parse_args()
    t = a.task
    spec_dir = os.path.join(paths.SPECS, str(t))
    out = os.path.join(paths.GOLD, str(t))
    if not a.skip_run:
        code = open(os.path.join(spec_dir, "reference.py")).read()
        data = os.path.join(paths.INPUTS, str(t), "dataset")
        run = run_with_capture(code, data, out, timeout=a.timeout, cell_id=f"gold-T{t:02d}")
        json.dump(run, open(os.path.join(out, "run.json"), "w"), indent=1)
        print(f"T{t:02d} reference success={run['success']} outputs={run['outputs']}")
        if not run["success"]:
            print(run["stderr_tail"][-1500:])
    res = evaluate(t, os.path.join(out, "pred_results"), os.path.join(out, "_task_root", "_snap"), out,
                   code_path=os.path.join(spec_dir, "reference.py"), stdout_path=os.path.join(out, "stdout.txt"))
    json.dump(res, open(os.path.join(out, "selftest.json"), "w"), indent=1)
    for r in res:
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}]{'' if r['required'] else ' (optional)'} {r['name']}: "
              f"value={r['value']} threshold={r['threshold']} {r['note'][:120]}")
    ok = checklib.verdict(res)
    print(f"T{t:02d} self-test (gold vs gold): {'PASS' if ok else 'SCORER CRASH' if ok is None else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
