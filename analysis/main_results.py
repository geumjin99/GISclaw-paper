#!/usr/bin/env python3
"""Main experiment and one-shot condition: accuracy, deliverable rate, paired differences.

    python3 analysis/main_results.py [--json FILE]

Reads results/main and results/oneshot and prints, for the fully gradable tasks:
  * the share of main-experiment runs that produced a deliverable and the share judged
    analytically correct, overall and per model over the four agent configurations;
  * accuracy with its 95% interval and the deliverable rate for every model and
    configuration (C0, SA H0, SA H1, SA H2, DA H2);
  * the paired differences H0 - C0, H1 - H0, H2 - H1, H2 - H0 and DA - SA per model,
    for all runs and for the runs whose replay was faithful;
  * the paired differences between each Claude model and each open-weight model;
  * per-task counts of correct runs and the one-shot runs that ended without code.
"""
import argparse
import collections

import stats

# column label -> (experiment, arch, tier)
COLUMNS = collections.OrderedDict([
    ("C0", ("oneshot", "SA", "C0")),
    ("SA H0", ("main", "SA", "H0")),
    ("SA H1", ("main", "SA", "H1")),
    ("SA H2", ("main", "SA", "H2")),
    ("DA H2", ("main", "DA", "H2")),
])
# paired differences reported per model: label -> (column a, column b), a - b
DIFFERENCES = collections.OrderedDict([
    ("H0 - C0", ("SA H0", "C0")),
    ("H1 - H0", ("SA H1", "SA H0")),
    ("H2 - H1", ("SA H2", "SA H1")),
    ("H2 - H0", ("SA H2", "SA H0")),
    ("DA - SA (H2)", ("DA H2", "SA H2")),
])
COMMERCIAL = [m for m in stats.MODELS if m not in stats.OPEN_WEIGHT]


def compute():
    rows = {"main": stats.load("main"), "oneshot": stats.load("oneshot")}

    def cell(model, column, faithful_only=False):
        exp, arch, tier = COLUMNS[column]
        sel = stats.select(rows[exp], model=model, arch=arch, tier=tier)
        return [r for r in sel if r["faithful"]] if faithful_only else sel

    main = stats.select(rows["main"])
    out = {
        "tasks": len(stats.TASKS),
        "headline": {"runs": len(main), "deliverable": stats.share(main, "deliverable"),
                     "correct": stats.share(main)},
        "per_model": {}, "table": {}, "differences": {}, "differences_faithful": {},
    }
    for model in stats.MODELS:
        sel = stats.select(rows["main"], model=model)
        out["per_model"][model] = {"runs": len(sel), "correct": stats.share(sel),
                                   "deliverable": stats.share(sel, "deliverable")}
        out["table"][model] = {}
        for column in COLUMNS:
            sel = cell(model, column)
            low, high = stats.bootstrap_ci(sel)
            out["table"][model][column] = {
                "runs": len(sel), "correct": stats.share(sel), "ci": [low, high],
                "deliverable": stats.share(sel, "deliverable"),
                "unfaithful": sum(not r["faithful"] for r in sel)}
        for key, faithful_only in (("differences", False), ("differences_faithful", True)):
            out[key][model] = {}
            for label, (a, b) in DIFFERENCES.items():
                d, low, high, n = stats.paired_difference(cell(model, a, faithful_only),
                                                          cell(model, b, faithful_only))
                out[key][model][label] = {"diff": d, "ci": [low, high], "tasks": n}

    between = []
    for column in list(COLUMNS)[1:]:
        for a in COMMERCIAL:
            for b in stats.OPEN_WEIGHT:
                d, low, high, _ = stats.paired_difference(cell(a, column), cell(b, column))
                between.append({"configuration": column, "a": a, "b": b, "diff": d, "ci": [low, high]})
    out["claude_minus_open_weight"] = between

    per_task = collections.Counter()
    solved_by = collections.defaultdict(set)
    for r in main:
        per_task[r["task"]] += r["correct"]
        if r["correct"]:
            solved_by[r["task"]].add(r["model"])
    runs_per_task = len(main) // len(stats.TASKS)
    out["per_task"] = {
        "runs_per_task": runs_per_task,
        "correct_runs": {t: per_task[t] for t in stats.TASKS},
        "always_solved": [t for t in stats.TASKS if per_task[t] == runs_per_task],
        "hardest": [{"task": t, "correct_runs": per_task[t], "solved_by": sorted(solved_by[t])}
                    for t in sorted(stats.TASKS, key=lambda t: (per_task[t], t))[:5]],
    }

    out["oneshot"] = {}
    for model in stats.MODELS:
        sel = cell(model, "C0")
        with_code = [r for r in sel if not r["no_code"]]
        out["oneshot"][model] = {"runs": len(sel), "without_code": len(sel) - len(with_code),
                                 "correct_with_code": stats.share(with_code)}
    return out


def _ci(ci):
    return f"[{stats.pct(ci[0])}, {stats.pct(ci[1])}]"


def report(res):
    h = res["headline"]
    print(f"Main experiment, {res['tasks']} fully gradable tasks, {h['runs']} runs")
    print(f"  produced a deliverable   {stats.pct(h['deliverable'])}%")
    print(f"  analytically correct     {stats.pct(h['correct'])}%")

    print("\nPer model over the four agent configurations")
    print(f"  {'model':18s} {'runs':>5s} {'correct %':>10s} {'deliverable %':>14s}")
    for model, label in stats.MODELS.items():
        m = res["per_model"][model]
        print(f"  {label:18s} {m['runs']:5d} {stats.pct(m['correct']):>10s} {stats.pct(m['deliverable']):>14s}")

    print("\nAccuracy % [95% interval] (deliverable %), per model and configuration")
    print(f"  {'model':18s} " + " ".join(f"{c:^26s}" for c in COLUMNS))
    for model, label in stats.MODELS.items():
        cells = []
        for column in COLUMNS:
            c = res["table"][model][column]
            cells.append(f"{stats.pct(c['correct'])} {_ci(c['ci'])} ({stats.pct(c['deliverable'])})")
        print(f"  {label:18s} " + " ".join(f"{c:^26s}" for c in cells))

    for key, title in (("differences", "Paired differences in pp [95% interval], * interval excludes zero"),
                       ("differences_faithful", "The same differences over runs with a faithful replay only")):
        print(f"\n{title}")
        print(f"  {'model':18s} " + " ".join(f"{d:^24s}" for d in DIFFERENCES))
        for model, label in stats.MODELS.items():
            cells = [stats.diff_text(d["diff"], *d["ci"]) for d in res[key][model].values()]
            print(f"  {label:18s} " + " ".join(f"{c:^24s}" for c in cells))

    print("\nRuns whose replay was not faithful, per model and configuration")
    print(f"  {'model':18s} " + " ".join(f"{c:>6s}" for c in COLUMNS))
    for model, label in stats.MODELS.items():
        print(f"  {label:18s} " + " ".join(f"{res['table'][model][c]['unfaithful']:6d}" for c in COLUMNS))

    b = res["claude_minus_open_weight"]
    print("\nClaude model minus open-weight model, same configuration (pp)")
    for x in b:
        print(f"  {x['configuration']:6s} {stats.SHORT[x['a']]:7s} - {stats.SHORT[x['b']]:8s} "
              f"{stats.diff_text(x['diff'], *x['ci'])}")
    print(f"  range {stats.pp(min(x['diff'] for x in b))} to {stats.pp(max(x['diff'] for x in b))}; "
          f"intervals excluding zero: {sum(stats.excludes_zero(*x['ci']) for x in b)} of {len(b)}")

    t = res["per_task"]
    print(f"\nCorrect runs per task (of {t['runs_per_task']})")
    print("  solved in every run: " + ", ".join(f"T{x:02d}" for x in t["always_solved"]))
    for x in t["hardest"]:
        print(f"  T{x['task']:02d} {x['correct_runs']:3d}  solved by "
              + (", ".join(stats.SHORT[m] for m in stats.MODELS if m in x["solved_by"]) or "none"))

    print("\nOne-shot condition (C0)")
    print(f"  {'model':18s} {'runs':>5s} {'without code':>13s} {'correct % of runs with code':>28s}")
    for model, label in stats.MODELS.items():
        o = res["oneshot"][model]
        print(f"  {label:18s} {o['runs']:5d} {o['without_code']:13d} {stats.pct(o['correct_with_code']):>28s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="also write the numbers to this file")
    args = ap.parse_args()
    res = compute()
    report(res)
    stats.write_json(res, args.json)


if __name__ == "__main__":
    main()
