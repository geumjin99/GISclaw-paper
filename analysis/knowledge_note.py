#!/usr/bin/env python3
"""Knowledge-note diagnostic: correct runs before and after adding one knowledge note.

    python3 analysis/knowledge_note.py [--json FILE]

Reads results/knowledge_note (tier H2+hint) and the H2 single-agent runs of results/main.
The unit of analysis is a model-task combination: its runs without the note are the three
main-experiment runs, its runs with the note are the three runs of the diagnostic.

For all combinations and per model the script prints the correct runs before and after,
the number of combinations that improved or got worse, the exact two-sided sign test, and
the mean change with a 95% bootstrap interval over the combinations (10,000 replicates).
The same is printed for two subsets: without the combinations that were already solved in
all three runs, and without the tasks whose note is close to a single check.
"""
import argparse

import stats

# tasks whose note comes close to what a single check tests (see the paper)
NEAR_CHECK_TASKS = (8, 24, 41, 44)


def load_units():
    """One entry per model-task combination: correct runs before and after, of how many.

    The combinations are sorted by model key and task; the bootstrap draws depend on
    this order.
    """
    main = stats.load("main")
    note = stats.load("knowledge_note")
    units = []
    for model, task in sorted({(r["model"], r["task"]) for r in note}):
        before = stats.select(main, [task], model=model, arch="SA", tier="H2")
        after = stats.select(note, [task], model=model)
        units.append({"model": model, "task": task,
                      "before": sum(r["correct"] for r in before), "runs_before": len(before),
                      "after": sum(r["correct"] for r in after), "runs_after": len(after)})
    return units


def summarize(units):
    change = [u["after"] / u["runs_after"] - u["before"] / u["runs_before"] for u in units]
    up, down = sum(c > 0 for c in change), sum(c < 0 for c in change)
    mean, low, high = stats.bootstrap_mean(change)
    return {"units": len(units), "before": sum(u["before"] for u in units),
            "after": sum(u["after"] for u in units), "runs": sum(u["runs_after"] for u in units),
            "up": up, "down": down, "sign_test_p": stats.sign_test(up, down),
            "mean_change": mean, "ci": [low, high]}


SUBSETS = (
    ("all combinations", lambda u: True),
    ("without combinations already solved in all runs", lambda u: u["before"] < u["runs_before"]),
    ("without tasks " + ", ".join(f"T{t:02d}" for t in NEAR_CHECK_TASKS),
     lambda u: u["task"] not in NEAR_CHECK_TASKS),
)


def compute():
    units = load_units()
    out = {"units": units, "subsets": {}}
    for name, keep in SUBSETS:
        kept = [u for u in units if keep(u)]
        block = {"all models": summarize(kept)}
        for model in stats.MODELS:
            sel = [u for u in kept if u["model"] == model]
            if sel:
                block[model] = summarize(sel)
        out["subsets"][name] = block
    return out


def report(res):
    for name, block in res["subsets"].items():
        print(f"{name}")
        print(f"  {'model':11s} {'units':>5s} {'correct runs':>15s} {'up':>3s} {'down':>4s} {'sign test p':>12s}"
              f"  mean change, pp [95% interval]")
        for key, s in block.items():
            runs = f"{s['before']} -> {s['after']} of {s['runs']}"
            print(f"  {stats.SHORT.get(key, key):11s} {s['units']:5d} {runs:>15s} {s['up']:3d} {s['down']:4d} "
                  f"{s['sign_test_p']:12.3f}  {stats.pp(s['mean_change'])} "
                  f"[{stats.pp(s['ci'][0])}, {stats.pp(s['ci'][1])}]")
        print()
    print("Correct runs per model-task combination (before -> after)")
    for u in res["units"]:
        print(f"  {stats.SHORT[u['model']]:8s} T{u['task']:02d}  {u['before']}/{u['runs_before']} -> "
              f"{u['after']}/{u['runs_after']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="also write the numbers to this file")
    args = ap.parse_args()
    res = compute()
    report(res)
    stats.write_json(res, args.json)


if __name__ == "__main__":
    main()
