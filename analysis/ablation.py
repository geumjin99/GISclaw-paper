#!/usr/bin/env python3
"""Ablations of the H2 single agent: task-message fields and GIS-layer components.

    python3 analysis/ablation.py [--json FILE]

Reads results/ablation and the H2 single-agent runs of results/main. For every model and
ablation it prints the change in accuracy against H2 (ablation - H2) in percentage points
with a 95% paired task-level bootstrap interval:
  * H2+nodd / H2+nodk: the dataset description or the domain-knowledge field removed
    from the task message;
  * H2-g1 ... H2-g6: one component of the GIS layer switched off;
  * H1 - H2 (whole GIS layer removed) and each component ablation against H1, for
    reference.
The field ablations are also given without the three tasks whose domain-knowledge field
states part of what the task requires; that restriction is a mean of per-task differences
bootstrapped over tasks with 10,000 replicates.
"""
import argparse
import collections

import stats

ABLATIONS = collections.OrderedDict([
    ("H2+nodd", "DD dataset description"),
    ("H2+nodk", "DK domain knowledge"),
    ("H2-g1", "G1 data-profile card"),
    ("H2-g2", "G2 operation guards"),
    ("H2-g3", "G3 deliverable checks"),
    ("H2-g4", "G4 library error notes"),
    ("H2-g5", "G5 analysis rules"),
    ("H2-g6", "G6 API lookup"),
])
FIELDS = ("H2+nodd", "H2+nodk")
# tasks whose domain-knowledge field also states part of the task's requirements
REQUIREMENT_IN_DK = (25, 31, 42)


def _task_differences(rows_a, rows_b, tasks):
    a, b = stats.by_task(rows_a), stats.by_task(rows_b)
    return [sum(a[t]) / len(a[t]) - sum(b[t]) / len(b[t]) for t in tasks]


def compute():
    main = stats.load("main")
    abl = stats.load("ablation")
    models = [m for m in stats.MODELS if any(r["model"] == m for r in abl)]
    reduced = [t for t in stats.TASKS if t not in REQUIREMENT_IN_DK]
    out = {"h2": {}, "ablations": {}, "whole_layer": {}, "against_h1": {}, "fields_reduced": {},
           "reduced_tasks": len(reduced)}
    for model in models:
        h2 = stats.select(main, model=model, arch="SA", tier="H2")
        h1 = stats.select(main, model=model, arch="SA", tier="H1")
        out["h2"][model] = stats.share(h2)
        out["ablations"][model], out["against_h1"][model], out["fields_reduced"][model] = {}, {}, {}
        for tier in ABLATIONS:
            sel = stats.select(abl, model=model, tier=tier)
            if not sel:
                continue
            d, low, high, n = stats.paired_difference(sel, h2)
            out["ablations"][model][tier] = {"correct": stats.share(sel), "runs": len(sel),
                                             "diff": d, "ci": [low, high], "tasks": n}
            if tier in FIELDS:
                d, low, high = stats.bootstrap_mean(_task_differences(
                    stats.select(sel, reduced), stats.select(h2, reduced), reduced))
                out["fields_reduced"][model][tier] = {"diff": d, "ci": [low, high]}
            else:
                d, low, high, _ = stats.paired_difference(sel, h1)
                out["against_h1"][model][tier] = {"diff": d, "ci": [low, high]}
        d, low, high, _ = stats.paired_difference(h1, h2)
        out["whole_layer"][model] = {"diff": d, "ci": [low, high]}
    return out


def _table(title, block, tiers, models):
    print(f"\n{title}")
    print(f"  {'removed':24s} " + " ".join(f"{stats.SHORT[m]:^24s}" for m in models))
    for tier in tiers:
        cells = [stats.diff_text(block[m][tier]["diff"], *block[m][tier]["ci"]) if tier in block[m] else "--"
                 for m in models]
        print(f"  {ABLATIONS[tier]:24s} " + " ".join(f"{c:^24s}" for c in cells))


def report(res):
    models = list(res["ablations"])
    print("H2 single-agent accuracy: "
          + ", ".join(f"{stats.SHORT[m]} {stats.pct(res['h2'][m])}%" for m in models))
    _table("Change in accuracy against H2, pp [95% interval], * interval excludes zero",
           res["ablations"], ABLATIONS, models)
    print(f"  {'whole GIS layer (H1-H2)':24s} "
          + " ".join(f"{stats.diff_text(res['whole_layer'][m]['diff'], *res['whole_layer'][m]['ci']):^24s}"
                     for m in models))
    _table(f"Field ablations without tasks {', '.join(f'T{t}' for t in REQUIREMENT_IN_DK)} "
           f"({res['reduced_tasks']} tasks)", res["fields_reduced"], FIELDS, models)
    _table("Component ablations against H1 (ablation - H1)", res["against_h1"],
           [t for t in ABLATIONS if t not in FIELDS], models)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="also write the numbers to this file")
    args = ap.parse_args()
    res = compute()
    report(res)
    stats.write_json(res, args.json)


if __name__ == "__main__":
    main()
