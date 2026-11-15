#!/usr/bin/env python3
"""Result figures drawn directly from the run records.

    python3 analysis/figures.py [--out DIR] [--only NAME ...]
                                [--notes DIR] [--embedding-model PATH]

Writes four PDF files to the output directory (default analysis/out):
  fig_main_accuracy.pdf     accuracy with 95% intervals and deliverable rate per model and
                            configuration (main_results.py)
  fig_ablation.pdf          change in accuracy when a task-message field or a GIS-layer
                            component is removed (ablation.py)
  fig_knowledge_note.pdf    correct runs before and after adding a knowledge note
                            (knowledge_note.py) and, when the knowledge notes are
                            available, the rank of each task's note in the offline
                            retrieval test (retrieval.py; dense and hybrid retrieval only
                            with --embedding-model)
  fig_variation.pdf         task-configuration cells by number of correct runs
                            (variation.py)
The numbers are computed by the scripts named in parentheses; the figures follow the
content of the corresponding figures of the paper, not their exact layout.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import ablation  # noqa: E402
import knowledge_note  # noqa: E402
import main_results  # noqa: E402
import retrieval  # noqa: E402
import stats  # noqa: E402
import variation  # noqa: E402

COLOR = {"claude-opus-5-5": "#0072B2", "claude-sonnet-5-5": "#56B4E9", "claude-haiku-5-5": "#009E73",
         "gptoss-120b": "#E69F00", "qwen38-27b": "#D55E00"}
MARKER = {"claude-opus-5-5": "o", "claude-sonnet-5-5": "s", "claude-haiku-5-5": "D",
          "gptoss-120b": "^", "qwen38-27b": "v"}
SHADES = ["#EEEEEE", "#BDD7E7", "#6BAED6", "#08519C"]
GREY = "#555555"


def _style():
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "legend.frameon": False, "pdf.fonttype": 42})


def fig_main_accuracy():
    res = main_results.compute()
    columns = list(main_results.COLUMNS)
    fig, axes = plt.subplots(1, len(stats.MODELS), figsize=(9.0, 2.8), sharey=True)
    for ax, (model, label) in zip(axes, stats.MODELS.items()):
        cells = [res["table"][model][c] for c in columns]
        x = range(len(columns))
        acc = [100 * c["correct"] for c in cells]
        ax.errorbar(x, acc, yerr=[[a - 100 * c["ci"][0] for a, c in zip(acc, cells)],
                                  [100 * c["ci"][1] - a for a, c in zip(acc, cells)]],
                    fmt=MARKER[model], color=COLOR[model], ms=5, lw=0.9, capsize=2)
        ax.plot(x, [100 * c["deliverable"] for c in cells], MARKER[model], ms=5, mfc="white",
                mec=GREY, mew=0.8, zorder=1)
        ax.axvline(0.5, color="#BBBBBB", lw=0.6, ls=(0, (3, 2)))
        ax.set_title(label, fontsize=8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([c.replace(" ", "\n") for c in columns])
        ax.set_xlim(-0.5, len(columns) - 0.5)
        ax.set_ylim(0, 104)
        ax.grid(axis="y", color="#E6E6E6", lw=0.5)
    axes[0].set_ylabel("Runs (%)")
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, ms=5, label="analytically correct (95% interval)"),
                        Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, ms=5, label="deliverable produced")],
               loc="lower center", ncol=2, fontsize=7.5)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.26, wspace=0.08)
    return fig


def fig_ablation():
    res = ablation.compute()
    models = list(res["ablations"])
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    y, ticks = 0.0, []
    rows = [(tier, label, res["ablations"]) for tier, label in ablation.ABLATIONS.items()]
    rows.append((None, "whole GIS layer (H1 - H2)", res["whole_layer"]))
    for tier, label, block in rows:
        present = [m for m in models if tier is None or tier in block[m]]
        if tier is None:
            present = [m for m in present if "H2-g1" in res["ablations"][m]]
        for j, model in enumerate(present):
            d = block[model] if tier is None else block[model][tier]
            yy = y + (j - (len(present) - 1) / 2) * 0.2
            low, high = 100 * d["ci"][0], 100 * d["ci"][1]
            ax.plot([low, high], [yy, yy], color=COLOR[model], lw=1.0)
            ax.plot(100 * d["diff"], yy, MARKER[model], ms=4.5, mec=COLOR[model], mew=0.9,
                    mfc=COLOR[model] if stats.excludes_zero(low, high) else "white")
        ticks.append((y, label))
        y += 1.0 if tier not in ablation.FIELDS else 1.2
    ax.axvline(0, color=GREY, lw=0.7)
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([lab for _, lab in ticks])
    ax.set_ylim(y - 0.4, -0.7)
    ax.set_xlabel("Change in accuracy when removed (pp)")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker=MARKER[m], color=COLOR[m], ms=4.5, lw=1.0, label=stats.SHORT[m]) for m in models]
    handles.append(Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, ms=4.5, label="interval includes 0"))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=7.5)
    fig.subplots_adjust(left=0.42, right=0.97, top=0.88, bottom=0.11)
    return fig


def _panel_before_after(ax, res):
    y, ticks = 0.0, []
    for model in stats.MODELS:
        units = [u for u in res["units"] if u["model"] == model]
        if not units:
            continue
        s = res["subsets"]["all combinations"][model]
        ax.text(-0.25, y, stats.SHORT[model], color=COLOR[model], fontweight="bold", va="center", fontsize=7.5)
        ax.text(3.25, y, f"{s['before']} to {s['after']} of {s['runs']}", ha="right", va="center", fontsize=7)
        y += 1
        for u in units:
            if u["after"] != u["before"]:
                ax.annotate("", xy=(u["after"], y), xytext=(u["before"], y),
                            arrowprops=dict(arrowstyle="-|>", lw=0.9, shrinkA=3, shrinkB=3, mutation_scale=7,
                                            color=COLOR[model] if u["after"] > u["before"] else "black",
                                            ls="-" if u["after"] > u["before"] else (0, (2, 1.5))))
            ax.plot(u["before"], y, "o", ms=5, mfc="white", mec=GREY, mew=0.8, zorder=3)
            ax.plot(u["after"], y, MARKER[model], ms=4, color=COLOR[model], zorder=4)
            ticks.append((y, f"T{u['task']:02d}"))
            y += 1
        y += 0.4
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([lab for _, lab in ticks], fontsize=7)
    ax.set_ylim(y - 0.6, -0.7)
    ax.set_xlim(-0.3, 3.3)
    ax.set_xticks(range(4))
    ax.set_xlabel("Correct runs of 3 (open: without note; filled: with note)")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


def _panel_ranks(ax, ret):
    pool = next(iter(ret["pools"].values()))
    styles = {"BM25": ("o", "black", "black"), "dense": ("s", "white", "black"), "hybrid": ("D", "#CC79A7", "#CC79A7")}
    methods = [m for m in styles if m in pool["ranks"]]
    tasks = sorted(pool["ranks"]["BM25"], key=lambda t: (min(pool["ranks"][m][t] for m in methods), t))
    for i, task in enumerate(tasks):
        ax.plot([1, pool["entries"]], [i, i], color="#E6E6E6", lw=0.5, zorder=0)
        for j, method in enumerate(methods):
            marker, face, edge = styles[method]
            ax.plot(pool["ranks"][method][task], i + (j - (len(methods) - 1) / 2) * 0.24, marker, ms=4,
                    mfc=face, mec=edge, mew=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.7, pool["entries"] * 1.2)
    ax.set_xticks([1, 3, 5, 10, 30, 100, pool["entries"]])
    ax.set_xticklabels(["1", "3", "5", "10", "30", "100", str(pool["entries"])])
    ax.minorticks_off()
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f"T{t:02d}" for t in tasks], fontsize=7)
    ax.set_ylim(len(tasks) - 0.5, -0.7)
    ax.set_xlabel(f"Rank of the task's own note among {pool['entries']} entries")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker=styles[m][0], ls="", mfc=styles[m][1], mec=styles[m][2], ms=4, label=m)
                       for m in methods], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(methods),
              fontsize=7, handletextpad=0.3, columnspacing=1.2)


def fig_knowledge_note(notes_dir, embedding_model):
    res = knowledge_note.compute()
    ret = None
    if os.path.isdir(notes_dir):
        ret = retrieval.compute(notes_dir, embedding_model)
    else:
        print(f"  {notes_dir} not found: the retrieval panel is left out")
    fig, axes = plt.subplots(1, 2 if ret else 1, figsize=(8.6 if ret else 4.3, 4.6), squeeze=False)
    _panel_before_after(axes[0][0], res)
    if ret:
        _panel_ranks(axes[0][1], ret)
        for ax, letter in zip(axes[0], "ab"):
            ax.set_title(letter, loc="left", fontweight="bold", fontsize=9)
    fig.subplots_adjust(left=0.07 if ret else 0.13, right=0.98, top=0.94, bottom=0.11, wspace=0.22)
    return fig


def fig_variation():
    res = variation.compute()
    fig, ax = plt.subplots(figsize=(4.6, 2.3))
    for i, model in enumerate(stats.MODELS):
        r = res[model]
        left = 0.0
        for k in range(4):
            width = 100 * r["correct_runs"][k] / r["cells"]
            ax.barh(i, width, left=left, height=0.68, color=SHADES[k])
            if width >= 6:
                ax.text(left + width / 2, i, str(r["correct_runs"][k]), ha="center", va="center", fontsize=7,
                        color="white" if k == 3 else "black")
            left += width
        ax.text(101.5, i, f"{stats.pct(r['mixed'])}%", va="center", fontsize=7.5)
    ax.text(101.5, -0.65, "mixed", va="bottom", fontsize=7, color=GREY)
    ax.set_yticks(range(len(stats.MODELS)))
    ax.set_yticklabels([stats.SHORT[m] for m in stats.MODELS])
    ax.set_ylim(len(stats.MODELS) - 0.45, -0.55)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Task-configuration cells (%)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.legend(handles=[Patch(color=SHADES[k], label=f"{k} of 3 correct") for k in range(4)],
               loc="upper center", ncol=4, fontsize=7, handlelength=1.0, columnspacing=1.0)
    fig.subplots_adjust(left=0.14, right=0.87, top=0.84, bottom=0.2)
    return fig


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(here, "out"), help="output directory (default analysis/out)")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    choices=["main_accuracy", "ablation", "knowledge_note", "variation"],
                    help="draw only these figures")
    ap.add_argument("--notes", default=retrieval.paths.NOTES, help="directory with the knowledge notes")
    ap.add_argument("--embedding-model", default=os.environ.get("GISCLAW_EMBEDDING_MODEL"),
                    help="local path of BAAI/bge-large-en-v1.5 for the retrieval panel")
    args = ap.parse_args()
    figures = {
        "main_accuracy": fig_main_accuracy,
        "ablation": fig_ablation,
        "knowledge_note": lambda: fig_knowledge_note(args.notes, args.embedding_model),
        "variation": fig_variation,
    }
    _style()
    os.makedirs(args.out, exist_ok=True)
    for name in args.only or figures:
        print(f"fig_{name}")
        fig = figures[name]()
        path = os.path.join(args.out, f"fig_{name}.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"  written to {path}")


if __name__ == "__main__":
    main()
