#!/usr/bin/env python3
"""Run-to-run variation: agreement of the three runs of a task under one configuration.

    python3 analysis/variation.py [--json FILE]

Reads results/main. A cell is one fully gradable task under one of the four agent
configurations of one model. The script prints, per model, the number of cells with 0, 1,
2 and 3 correct runs and the share of cells with mixed results (1 or 2 correct).
"""
import argparse
import collections

import stats


def compute():
    main = stats.load("main")
    out = {}
    for model in stats.MODELS:
        counts = collections.Counter()
        for arch, tier in stats.CONFIGS:
            groups = stats.by_task(stats.select(main, model=model, arch=arch, tier=tier))
            counts.update(sum(v) for v in groups.values())
        cells = sum(counts.values())
        out[model] = {"cells": cells, "correct_runs": {k: counts[k] for k in range(4)},
                      "mixed": (counts[1] + counts[2]) / cells}
    return out


def report(res):
    print("Task-configuration cells by number of correct runs out of three")
    print(f"  {'model':18s} {'cells':>5s} {'0':>5s} {'1':>5s} {'2':>5s} {'3':>5s} {'mixed %':>8s}")
    for model, label in stats.MODELS.items():
        r = res[model]
        print(f"  {label:18s} {r['cells']:5d} " + " ".join(f"{r['correct_runs'][k]:5d}" for k in range(4))
              + f" {stats.pct(r['mixed']):>8s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="also write the numbers to this file")
    args = ap.parse_args()
    res = compute()
    report(res)
    stats.write_json(res, args.json)


if __name__ == "__main__":
    main()
