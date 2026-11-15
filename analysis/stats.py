"""Shared helpers for the analysis scripts: run records and resampling statistics.

A run is one directory results/<experiment>/<tier>/<arch>/<model>/run<k>/T<id>/ holding
cell_result.json (the run record) and cell_eval.json (the verdict). `load` turns the runs
of one experiment into a list of flat rows.

Statistics
  * Accuracy is the share of runs judged analytically correct. A run without a verdict
    counts as not correct.
  * Confidence intervals are 95% percentile intervals of a task-level cluster bootstrap:
    tasks are resampled with replacement and every run of a drawn task is kept, so the
    repeated runs of one task are not treated as independent.
  * A difference between two conditions is paired: each replicate draws the tasks once
    and uses the same draw for both conditions (tasks present in both only).
  * The sign test is the exact two-sided binomial test on the units that changed.

This module is imported by the other scripts and prints nothing itself.
"""
import collections
import glob
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "harness"))
import paths  # noqa: E402

# Task 30 is only partially gradable; it is left out of every accuracy figure.
PARTIALLY_GRADABLE = frozenset({30})
ALL_TASKS = tuple(range(1, 51))
TASKS = tuple(t for t in ALL_TASKS if t not in PARTIALLY_GRADABLE)

# model key in the records -> name used in the paper, in the paper's order
MODELS = collections.OrderedDict([
    ("claude-opus-5-5", "Claude Opus 5.5"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
    ("claude-haiku-5-5", "Claude Haiku 5.5"),
    ("gptoss-120b", "gpt-oss-120b"),
    ("qwen38-27b", "Qwen3.8-27B"),
])
SHORT = {"claude-opus-5-5": "Opus", "claude-sonnet-5-5": "Sonnet", "claude-haiku-5-5": "Haiku",
         "gptoss-120b": "gpt-oss", "qwen38-27b": "Qwen"}
OPEN_WEIGHT = ("gptoss-120b", "qwen38-27b")

# the four agent configurations of the main experiment, as (arch, tier)
CONFIGS = (("SA", "H0"), ("SA", "H1"), ("SA", "H2"), ("DA", "H2"))

REPLICATES = 2000
REPLICATES_SMALL = 10000   # perturbation test and knowledge-note diagnostic
SEED = 0


def load(experiment, results_dir=None):
    """Rows of one experiment, one per run, sorted by (tier, arch, model, task, run)."""
    base = os.path.join(results_dir or paths.RESULTS, experiment)
    rows = []
    for path in glob.glob(os.path.join(base, "*", "*", "*", "run*", "T*", "cell_result.json")):
        with open(path, encoding="utf-8") as fh:
            res = json.load(fh)
        eval_path = os.path.join(os.path.dirname(path), "cell_eval.json")
        ev = {}
        if os.path.exists(eval_path):
            with open(eval_path, encoding="utf-8") as fh:
                ev = json.load(fh)
        llm = res.get("llm_stats") or {}
        rows.append({
            "experiment": res["experiment"], "tier": res["tier"], "arch": res["arch"],
            "model": res["model"], "run": res["run"], "task": res["task"],
            "correct": ev.get("analytically_correct") is True,
            "faithful": ev.get("replay_faithful") is not False,
            "deliverable": bool(res.get("deliverable_produced")),
            "no_code": bool(res.get("oneshot_no_code")),
            "elapsed_s": res.get("elapsed_s") or 0.0,
            "rounds": res.get("rounds") or 0,
            "calls": llm.get("calls") or 0,
            "input_tokens": ((llm.get("prompt_tokens") or 0) + (llm.get("cached_tokens") or 0)
                             + (llm.get("cache_write_tokens") or 0)),
            "output_tokens": llm.get("completion_tokens") or 0,
        })
    if not rows:
        raise SystemExit(f"no run records found under {base}")
    rows.sort(key=lambda r: (r["tier"], r["arch"], r["model"], r["task"], r["run"]))
    return rows


def select(rows, tasks=TASKS, **equal):
    """Rows whose task is in `tasks` and whose fields equal the given values."""
    tasks = set(tasks)
    return [r for r in rows if r["task"] in tasks and all(r[k] == v for k, v in equal.items())]


def share(rows, field="correct"):
    return sum(bool(r[field]) for r in rows) / len(rows)


def by_task(rows, field="correct"):
    out = collections.defaultdict(list)
    for r in rows:
        out[r["task"]].append(bool(r[field]))
    return out


def _interval(values):
    values = sorted(values)
    n = len(values)
    return values[int(0.025 * n)], values[int(0.975 * n) - 1]


def bootstrap_ci(rows, replicates=REPLICATES, seed=SEED, field="correct"):
    """95% interval of the share of `field` over `rows`, resampling tasks."""
    groups = by_task(rows, field)
    tasks = sorted(groups)
    rng = random.Random(seed)
    stats = []
    for _ in range(replicates):
        vals = [v for t in (rng.choice(tasks) for _ in tasks) for v in groups[t]]
        stats.append(sum(vals) / len(vals))
    return _interval(stats)


def paired_difference(rows_a, rows_b, replicates=REPLICATES, seed=SEED, field="correct", rng=None):
    """share(a) - share(b) on the tasks present in both, with its 95% interval.

    Returns (difference, low, high, number of tasks). A generator passed as `rng` is used
    instead of a new one seeded with `seed`, so that several comparisons can share one
    random stream.
    """
    a, b = by_task(rows_a, field), by_task(rows_b, field)
    tasks = sorted(set(a) & set(b))

    def stat(draw):
        va = [v for t in draw for v in a[t]]
        vb = [v for t in draw for v in b[t]]
        return sum(va) / len(va) - sum(vb) / len(vb)

    rng = rng or random.Random(seed)
    stats = [stat([rng.choice(tasks) for _ in tasks]) for _ in range(replicates)]
    low, high = _interval(stats)
    return stat(tasks), low, high, len(tasks)


def bootstrap_mean(values, replicates=REPLICATES_SMALL, seed=SEED):
    """Mean of `values` (one per unit) with a 95% interval, resampling the units."""
    values = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (replicates, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test(up, down):
    """Exact two-sided sign test for `up` increases and `down` decreases (ties dropped)."""
    n, k = up + down, max(up, down)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def excludes_zero(low, high):
    return low > 0 or high < 0


def quartiles(values):
    """(median, first quartile, third quartile) with linear interpolation."""
    v = sorted(values)

    def q(p):
        pos = p * (len(v) - 1)
        i = int(math.floor(pos))
        j = min(i + 1, len(v) - 1)
        return v[i] + (v[j] - v[i]) * (pos - i)

    return q(0.5), q(0.25), q(0.75)


def pp(x):
    """A difference of shares as signed percentage points."""
    return f"{round(100 * x, 1) + 0.0:+.1f}"


def pct(x):
    return f"{100 * x:.1f}"


def diff_text(diff, low, high):
    star = "*" if excludes_zero(low, high) else ""
    return f"{pp(diff)} [{pp(low)}, {pp(high)}]{star}"


def write_json(obj, path):
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)
