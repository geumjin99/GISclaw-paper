#!/usr/bin/env python3
"""Token use, run time and number of rounds in the main experiment.

    python3 analysis/cost_and_tokens.py [--json FILE]

Reads results/main and prints:
  * token use per run (median and interquartile range, thousands of tokens, summed over
    all model calls of a run) for the single agent (H0, H1 and H2 pooled) and the dual
    agent, on the fully gradable tasks. Input tokens are prompt tokens plus the prompt
    tokens read from or written to the provider's cache; output tokens are completion
    tokens, which include reasoning tokens;
  * median run time and median number of rounds per model and configuration, on the fully
    gradable tasks;
  * total run time per model over all runs of the main experiment, all tasks included.
"""
import argparse

import stats


def _median(values):
    return stats.quartiles(values)[0]


def compute():
    main = stats.load("main")
    out = {"tokens": {}, "per_configuration": {}, "total_run_time_h": {}}
    for model in stats.MODELS:
        out["tokens"][model] = {}
        for arch in ("SA", "DA"):
            sel = stats.select(main, model=model, arch=arch)
            out["tokens"][model][arch] = {
                "runs": len(sel),
                "input_k": [q / 1000 for q in stats.quartiles([r["input_tokens"] for r in sel])],
                "output_k": [q / 1000 for q in stats.quartiles([r["output_tokens"] for r in sel])]}
        out["per_configuration"][model] = {}
        for arch, tier in stats.CONFIGS:
            sel = stats.select(main, model=model, arch=arch, tier=tier)
            out["per_configuration"][model][f"{arch} {tier}"] = {
                "runs": len(sel), "median_elapsed_s": _median([r["elapsed_s"] for r in sel]),
                "median_rounds": _median([r["rounds"] for r in sel])}
        every = stats.select(main, stats.ALL_TASKS, model=model)
        out["total_run_time_h"][model] = {"runs": len(every),
                                          "hours": sum(r["elapsed_s"] for r in every) / 3600}
    return out


def _iqr(q, digits):
    return f"{q[0]:.{digits}f} [{q[1]:.{digits}f}, {q[2]:.{digits}f}]"


def report(res):
    print("Token use per run, thousands of tokens: median [interquartile range]")
    print(f"  {'model':18s} {'SA input':^16s} {'SA output':^18s} {'DA input':^16s} {'DA output':^18s} {'runs SA/DA':>10s}")
    for model, label in stats.MODELS.items():
        t = res["tokens"][model]
        print(f"  {label:18s} {_iqr(t['SA']['input_k'], 0):^16s} {_iqr(t['SA']['output_k'], 1):^18s} "
              f"{_iqr(t['DA']['input_k'], 0):^16s} {_iqr(t['DA']['output_k'], 1):^18s} "
              f"{str(t['SA']['runs']) + '/' + str(t['DA']['runs']):>10s}")

    configs = [f"{arch} {tier}" for arch, tier in stats.CONFIGS]
    for key, title, fmt in (("median_elapsed_s", "Median run time (s)", "{:.0f}"),
                            ("median_rounds", "Median number of rounds", "{:.0f}")):
        print(f"\n{title}")
        print(f"  {'model':18s} " + " ".join(f"{c:>7s}" for c in configs))
        for model, label in stats.MODELS.items():
            print(f"  {label:18s} " + " ".join(
                f"{fmt.format(res['per_configuration'][model][c][key]):>7s}" for c in configs))

    print("\nTotal run time over all main-experiment runs")
    for model, label in stats.MODELS.items():
        t = res["total_run_time_h"][model]
        print(f"  {label:18s} {t['runs']:4d} runs {t['hours']:7.1f} h")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="also write the numbers to this file")
    args = ap.parse_args()
    res = compute()
    report(res)
    stats.write_json(res, args.json)


if __name__ == "__main__":
    main()
