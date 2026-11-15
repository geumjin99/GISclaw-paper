#!/usr/bin/env python3
"""Perturbation test: perturbed task variants against the unperturbed tasks.

    python3 analysis/perturbation.py [--json FILE]

Reads results/perturbation, results/perturbation_rerun and the H2 single-agent runs of
results/main. A perturbed variant has the id 100 * type + base task, with type 1 = inputs
reprojected to another CRS, 2 = attribute fields renamed, 3 = instruction reworded.

Three comparisons are printed, each as correct runs in both conditions and the difference
in percentage points with a 95% paired task-level bootstrap interval (10,000 replicates):
  1. perturbed variants against the main-experiment runs of the same tasks;
  2. perturbed variants against the unperturbed rerun of the same tasks (the table of the
     perturbation test in the paper);
  3. the unperturbed rerun against the main-experiment runs (run-to-run variation).
The comparisons are computed in this order from one random stream.
"""
import argparse
import collections
import random

import stats

TYPES = collections.OrderedDict([(1, "Reprojected inputs"), (2, "Renamed fields"), (3, "Reworded instruction")])
BOOTSTRAP_SEED = 20261102


def compute():
    main = stats.load("main")
    rerun = stats.load("perturbation_rerun")
    pert = stats.load("perturbation")
    models = [m for m in stats.MODELS if any(r["model"] == m for r in pert)]
    base_tasks = sorted({r["task"] for r in rerun})
    rng = random.Random(BOOTSTRAP_SEED)

    def perturbed(kind, model):
        return [dict(r, task=r["task"] % 100) for r in pert
                if r["task"] // 100 == kind and r["model"] == model]

    def reference(name, model, tasks):
        if name == "rerun":
            return stats.select(rerun, tasks, model=model)
        return stats.select(main, tasks, model=model, arch="SA", tier="H2")

    def compare(a, b):
        d, low, high, n = stats.paired_difference(a, b, stats.REPLICATES_SMALL, rng=rng)
        return {"tasks": n, "a_correct": sum(r["correct"] for r in a), "a_runs": len(a),
                "b_correct": sum(r["correct"] for r in b), "b_runs": len(b), "diff": d, "ci": [low, high]}

    out = {"base_tasks": base_tasks, "variants": len({r["task"] for r in pert})}
    for name in ("main", "rerun"):
        block = []
        for kind in TYPES:
            for model in models:
                a = perturbed(kind, model)
                b = reference(name, model, {r["task"] for r in a})
                block.append(dict(compare(a, b), perturbation=TYPES[kind], model=model))
        out[f"perturbed_vs_{name}"] = block
    out["rerun_vs_main"] = [dict(compare(reference("rerun", m, base_tasks), reference("main", m, base_tasks)),
                                 model=m) for m in models]
    return out


def _rows(block, a_name, b_name, labelled=True):
    print(f"  {'perturbation' if labelled else '':22s} {'model':8s} {'tasks':>5s} {a_name:>10s} {b_name:>10s}  difference")
    for x in block:
        print(f"  {x.get('perturbation', ''):22s} {stats.SHORT[x['model']]:8s} {x['tasks']:5d} "
              f"{str(x['a_correct']) + '/' + str(x['a_runs']):>10s} "
              f"{str(x['b_correct']) + '/' + str(x['b_runs']):>10s}  "
              f"{stats.diff_text(x['diff'], *x['ci'])}")


def report(res):
    print(f"{len(res['base_tasks'])} base tasks ({', '.join(f'T{t:02d}' for t in res['base_tasks'])}), "
          f"{res['variants']} perturbed variants; H2 single agent")
    print("\n1. Perturbed variants against the main-experiment runs of the same tasks")
    _rows(res["perturbed_vs_main"], "perturbed", "main")
    print("\n2. Perturbed variants against the unperturbed rerun of the same tasks")
    _rows(res["perturbed_vs_rerun"], "perturbed", "rerun")
    print("\n3. Unperturbed rerun against the main-experiment runs")
    _rows(res["rerun_vs_main"], "rerun", "main", labelled=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="also write the numbers to this file")
    args = ap.parse_args()
    res = compute()
    report(res)
    stats.write_json(res, args.json)


if __name__ == "__main__":
    main()
