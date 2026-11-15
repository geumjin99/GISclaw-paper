#!/usr/bin/env python3
"""Score finished runs for analytical correctness.

    python3 harness/score.py --exp main [--tasks 1,8] [--jobs 4] [--force]

A run is one directory results/<experiment>/<tier>/<arch>/<model>/run<k>/T<id>/ written by
run_cell.py: cell_result.json (the run record), exec_log.jsonl (every code block the agent
executed, in order, with its outcome), rounds.jsonl (the agent's steps and the observations it
received) and <model>/ (the files the agent wrote to pred_results/). The verdict is written
to <run>/cell_eval.json. A run that already has one is skipped unless --force is given.

How a run is scored
  1. Replay. Every executed block, failed ones included (they can leave partial state), is
     re-executed in the original order in a fresh container, with data capture (capture.py)
     and a fixed random seed. Each block gets three times the step limit of the run, because
     the capture slows the code down.
  2. Checks. The task's checks (benchmark/specs/<id>/checks.py) compare the run with the
     reference solution's outputs (benchmark/gold/<id>). Checks on output files read the
     files of the run itself; checks on the data behind the outputs read the snapshots
     captured during the replay; checks on the code read the text of the successful blocks
     plus the statements of failed blocks that completed before the error.
  3. Verdict. `analytically_correct` is true when the run has at least one required check
     and every required check passes. Checks that are not required (for example checks of
     how a figure presents the data) are reported and do not affect the verdict. An
     unverifiable check is a required check that the captured evidence cannot decide: it
     does not pass, because a correct verdict needs positive evidence. A crash of the checks
     themselves gives no verdict (`analytically_correct` null, `scorer_error` true).

Rules applied to every task (checklib.py, "uniform rules")
  * Every geographic file that the task's contract (benchmark/specs/<id>/contract.json)
    declares as a deliverable must carry a CRS; this is one required check.
  * An output file that is only a copy of an input dataset is not a result: it is hidden
    from the checks and listed in an informational entry.

Emergency files
  When an agent ends without having saved its results, the harness saves what is in memory
  under names starting with "emergency_". These files are scored like any other output;
  `emergency_files` lists them and `correct_without_emergency` tells whether the verdict
  holds when they are hidden.

Faithful replay
  The replay is faithful when every block succeeds or fails as it did in the run, no
  replayed block is cut off (time limit, memory limit, container crash) and the replay
  container is never restarted with a partial state restore. `replay_faithful` and
  `replay_unfaithful_reasons` record this; an unfaithful replay still gets a verdict. The one
  exception is a replay broken by a container failure that the run did not have (a block
  killed, or a failed state restore): it is repeated once with twice the memory, and if it
  fails again the run gets no verdict (`replay_container_failure`).

Fields of cell_eval.json
  status                 "ok", "no_code" (nothing was executed: only the output files are
                         checked), "not_scored" (the run did not finish normally),
                         "no_spec" (the task has no checks) or "eval_error"
  analytically_correct   true / false / null (no verdict)
  scorer_error           true when the checks crashed or the replay container failed
  checks                 one entry per check: name, passed, value, threshold, required, note
  replay                 success, outputs, same_outputs_as_original, blocks_ok,
                         block_outcome, block_mismatches, events, step_timeout_s
  replay_faithful, replay_unfaithful_reasons, replay_container_failure
  emergency_files, correct_without_emergency
"""
import argparse
import ast
import collections
import glob
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from capture import run_with_capture  # noqa: E402
from build_gold import evaluate  # noqa: E402
import checklib  # noqa: E402

# Names the harness reserves next to the agent's outputs for its own records. Files with these
# names are not results and are never scored.
NON_AGENT_FILES = {"code.py", "execution_log.txt", "report.md"}
EMERGENCY_PREFIX = "emergency_"
ABNORMAL_OUTCOMES = ("timeout_interrupted", "timeout_killed", "oom", "crash", "exception")
KILL_OUTCOMES = ("oom", "crash", "timeout_killed")   # the container was killed or died

REPLAY_TIMEOUT_FACTOR = 3   # replay step limit = step limit of the run x this
MEMORY = None               # --memory: memory limit of the replay container
SKIP_KILLED = False         # --skip-killed


def agent_outputs_copy(model_out, dest):
    """Copy of the agent's output folder without the harness-written files."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    if model_out and os.path.isdir(model_out):
        for n in os.listdir(model_out):
            if n in NON_AGENT_FILES:
                continue
            src = os.path.join(model_out, n)
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(dest, n))
    return dest


def replay_faithfulness(entries, run):
    """(faithful, reasons, mismatching blocks). Faithful = every block succeeded/failed as in
    the run AND no replay block was cut off (timeout / OOM / crash) AND the replay container
    was never restarted with a partial state restore. A cut-off block leaves a partial state
    that need not match the one the run had."""
    reasons = []
    orig = [bool(e.get("success")) for e in entries]
    got = run.get("block_success", [])
    diff = [i for i, (a, b) in enumerate(zip(orig, got)) if a != b]
    if diff:
        reasons.append(f"block success differs at {diff[:10]}")
    if len(orig) != len(got):
        reasons.append(f"{len(got)} of {len(orig)} blocks replayed")
    outc = run.get("block_outcome") or []
    cut = [i for i, o in enumerate(outc) if o in ABNORMAL_OUTCOMES]
    if cut:
        reasons.append(f"replay block(s) {cut[:10]} cut off ({outc[cut[0]]})")
    evs = [e.get("event") for e in run.get("events", [])]
    if "replay_partial" in evs:
        reasons.append("container restarted with a partial state restore (replay_partial)")
    if "restart_start_failed" in evs:
        reasons.append("container restart failed")
    return not reasons, reasons, diff


def replay_container_failure(entries, run, orig_events):
    """Reasons why the replay's state is broken by a container failure that the run did not
    have: a block killed (OOM / crash / kill after timeout) whose outcome in the run was
    different, or a container restart whose state restore failed partly although the run's
    did not. Such a replay cannot judge the agent: the run gets no verdict. Other replay
    mismatches keep their verdict."""
    reasons = []
    outc = run.get("block_outcome") or []
    for i, o in enumerate(outc):
        # oom / crash are the same event seen differently (the container died): flag only a
        # kill where the block of the run was not killed
        if o in KILL_OUTCOMES and i < len(entries) and entries[i].get("outcome") not in KILL_OUTCOMES:
            reasons.append(f"replay block {i} {o} (original: {entries[i].get('outcome')})")
    oev = {e.get("event") for e in orig_events or []}
    for e in run.get("events", []):
        ev = e.get("event")
        if ev in ("replay_partial", "restart_start_failed") and ev not in oev:
            reasons.append(f"replay {ev}" + (f" {e.get('failed_blocks')}" if e.get("failed_blocks") else ""))
    return reasons


_MODULE_LINE = re.compile(r'File "<string>", line (\d+), in <module>')


def failed_block_prefix(code, observation):
    """The top-level statements of a failed block that ran to completion before the error:
    everything before the top-level statement holding the line where the exception left the
    block (first `File "<string>", line N, in <module>` frame of the last traceback in the
    agent's observation). None when the line cannot be located (no traceback, truncated
    output, syntax error, OOM, blocked import): then nothing of the block is used."""
    if not observation or "Traceback (most recent call last)" not in observation:
        return None
    tail = observation[observation.rfind("Traceback (most recent call last)"):]
    m = _MODULE_LINE.search(tail)
    if not m:
        return None
    n = int(m.group(1))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    lines = code.splitlines()
    if n < 1 or n > len(lines):
        return None
    for st in tree.body:
        start = min([st.lineno] + [d.lineno for d in getattr(st, "decorator_list", [])])
        if start <= n <= st.end_lineno:
            pre = "\n".join(lines[:start - 1])
            return pre if pre.strip() else ""
    return None


def _last_line(s):
    return next((l.strip() for l in reversed((s or "").strip().splitlines()) if l.strip()), "")


def checked_code_text(cell_dir, entries, replay_tb=None):
    """The code text the checks read: the successful blocks, plus the statements of failed
    blocks that completed before the error (an assignment made before the failing statement
    took effect), in execution order. The failing statement and everything after it are left
    out. `replay_tb`: per block, the full traceback of the scoring replay; used for a block
    that failed in both runs when the agent's observation (tail-truncated stderr) cannot
    locate the failing line. Returns (text, number of failed blocks that contributed)."""
    obs = collections.defaultdict(collections.deque)
    try:
        for l in open(os.path.join(cell_dir, "rounds.jsonl")):
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("action") == "execute":
                obs[(r.get("args") or {}).get("code")].append(r.get("observation"))
    except (OSError, ValueError):
        obs = None
    parts, used = [], 0
    for i, e in enumerate(entries):
        o = None
        if obs is not None and obs.get(e["code"]):
            o = obs[e["code"]].popleft()
        if e.get("success"):
            parts.append(e["code"])
        elif obs is not None:
            pre = failed_block_prefix(e["code"], o)
            if pre is None and replay_tb and i < len(replay_tb) and replay_tb[i]:
                # only when the replay failed with the same error as the run (same final traceback
                # line), so a replay failing further down cannot credit statements the run never ran
                last = _last_line(replay_tb[i])
                if last and last in (o or ""):
                    pre = failed_block_prefix(e["code"], replay_tb[i])
            if pre:
                parts.append(pre)
                used += 1
    return "\n\n".join(parts), used


def _write(rec, out_path):
    json.dump(rec, open(out_path, "w"), indent=1)
    return rec


def eval_cell(cell_dir, force=False):
    out_path = os.path.join(cell_dir, "cell_eval.json")
    if os.path.exists(out_path) and not force:
        return json.load(open(out_path))
    res = json.load(open(os.path.join(cell_dir, "cell_result.json")))
    t = res["task"]
    rec = {"cell_id": res["cell_id"], "task": t}
    if res.get("status") != "done":
        # not a valid run (model server failure, sandbox failure, crash...): it must not count
        # as an incorrect analysis
        rec.update(status="not_scored", analytically_correct=None)
        return _write(rec, out_path)
    if not os.path.exists(os.path.join(paths.SPECS, str(t), "checks.py")):
        rec.update(status="no_spec", analytically_correct=None)
        return _write(rec, out_path)
    rp = os.path.join(cell_dir, "replay")
    agent_dir = agent_outputs_copy(os.path.join(cell_dir, res["model"]),
                                   os.path.join(cell_dir, "replay_agent_outputs"))
    log = os.path.join(cell_dir, "exec_log.jsonl")
    entries = [json.loads(l) for l in open(log) if l.strip()] if os.path.exists(log) else []
    if not entries:
        eval_args = dict(pred_snap=None)
        rec.update(status="no_code", checks=evaluate(t, agent_dir, None))
    else:
        cfg = res.get("config") or {}
        replay_timeout = int(int(cfg.get("step_timeout_s", 300)) * REPLAY_TIMEOUT_FACTOR)
        blocks = [e["code"] for e in entries]
        code_txt = os.path.join(cell_dir, "replay_agent_code.py")
        txt, n_partial = checked_code_text(cell_dir, entries)
        with open(code_txt, "w") as f:
            f.write(txt)
        skip = None
        if SKIP_KILLED:
            skip = {i: e.get("outcome") for i, e in enumerate(entries) if e.get("outcome") in KILL_OUTCOMES}
        mem = MEMORY or res.get("container_memory")
        for attempt in (1, 2):
            run = run_with_capture("", os.path.join(paths.INPUTS, str(t), "dataset"), rp,
                                   timeout=replay_timeout, memory=mem,
                                   cpus=str(res.get("container_cpus", 2)), image=cfg.get("image"),
                                   cell_id="replay-" + res["cell_id"], blocks=blocks, skip=skip)
            json.dump(run, open(os.path.join(rp, "run.json"), "w"), indent=1)
            rec["replay"] = {k: run[k] for k in ("success", "outputs", "events", "blocks_ok")}
            rec["replay"]["block_outcome"] = run.get("block_outcome")
            rec["replay"]["step_timeout_s"] = replay_timeout
            rec["replay"]["same_outputs_as_original"] = sorted(run["outputs"]) == sorted(res.get("outputs", []))
            faithful, reasons, diff = replay_faithfulness(entries, run)
            rec["replay"]["block_mismatches"] = diff
            rec["replay_faithful"] = faithful
            rec["replay_unfaithful_reasons"] = reasons
            rec["replay_container_failure"] = [] if faithful else \
                replay_container_failure(entries, run, res.get("sandbox_events"))
            if not rec["replay_container_failure"] or attempt == 2:
                break
            # the capture adds memory: replay once more with twice the container memory before
            # giving up on a verdict
            try:
                mem = f"{int(float(str(mem).rstrip('gG')) * 2)}g" if mem else "32g"
            except ValueError:
                mem = "32g"
            shutil.rmtree(rp, ignore_errors=True)
        # the checked code text again, now with the replay's full tracebacks (blocks that
        # failed in both runs only)
        tb = run.get("block_traceback") or []
        tb = [x if (i < len(entries) and not entries[i].get("success")) else None for i, x in enumerate(tb)]
        txt2, n2 = checked_code_text(cell_dir, entries, replay_tb=tb)
        if n2 != n_partial:
            with open(code_txt, "w") as f:
                f.write(txt2)
        # Files the agent wrote under output/ instead of pred_results/ (some instructions ask
        # for that folder); run_cell.py keeps a copy in extra_outputs/.
        extra = os.path.join(cell_dir, "extra_outputs", "output")
        if os.path.isdir(extra):
            for root, _, files in os.walk(extra):
                for n in files:
                    dst = os.path.join(agent_dir, n)
                    if not os.path.exists(dst):  # pred_results/ wins on a name clash
                        shutil.copy2(os.path.join(root, n), dst)
        # file checks on the files of the run; data checks on the replay snapshots
        eval_args = dict(pred_snap=os.path.join(rp, "_task_root", "_snap"), code_path=code_txt,
                         stdout_path=os.path.join(rp, "stdout.txt"))
        rec["checks"] = evaluate(t, agent_dir, **eval_args)
        if rec.get("replay_container_failure"):
            # the replay's state was broken by a container failure the run did not have: no
            # verdict either way
            rec["checks"].append(checklib.scorer_error_result(
                "replay container failure: " + "; ".join(rec["replay_container_failure"])[:300]))
        rec["status"] = "ok"
    # a crash of the task's checks is a scorer error: no verdict (None); the analysis scripts
    # count such a run as not correct
    rec["scorer_error"] = checklib.has_scorer_error(rec.get("checks", []))
    rec["analytically_correct"] = checklib.verdict(rec.get("checks", []))
    # emergency files are scored but counted apart: record them, and whether the verdict
    # holds without them
    emg = sorted(n for n in os.listdir(agent_dir) if n.startswith(EMERGENCY_PREFIX)) if os.path.isdir(agent_dir) else []
    rec["emergency_files"] = emg
    if emg:
        if rec["analytically_correct"]:
            alt = evaluate(t, agent_dir, exclude_prefixes=(EMERGENCY_PREFIX,), **eval_args)
            rec["correct_without_emergency"] = checklib.verdict(alt)
        elif rec["scorer_error"]:
            rec["correct_without_emergency"] = None
        else:
            rec["correct_without_emergency"] = False
    return _write(rec, out_path)


def eval_cell_safe(cell_dir, force=False):
    try:
        return eval_cell(cell_dir, force)
    except Exception:  # one broken run must not stop the batch or go unrecorded
        import traceback
        rec = {"cell_id": os.path.relpath(cell_dir, paths.RESULTS), "status": "eval_error",
               "analytically_correct": None, "error": traceback.format_exc()[-1500:]}
        try:
            json.dump(rec, open(os.path.join(cell_dir, "cell_eval.json"), "w"), indent=1)
        except OSError:
            pass
        return rec


def main():
    ap = argparse.ArgumentParser(description="Score finished runs for analytical correctness.")
    ap.add_argument("--exp", required=True, help="experiment name: the folder under results/")
    ap.add_argument("--tasks", help="comma-separated task ids (default: all)")
    ap.add_argument("--jobs", type=int, default=4, help="runs scored in parallel (default %(default)s)")
    ap.add_argument("--force", action="store_true", help="score again runs that already have a cell_eval.json")
    ap.add_argument("--cells", help="comma-separated run folders under results/<exp>, "
                    "i.e. <tier>/<arch>/<model>/run<k>/T<id>")
    ap.add_argument("--memory", help="memory limit of the replay container, e.g. 64g (default: the limit "
                    "the run had). The capture keeps copies of the data in memory, so a replay can need "
                    "more than the run did; the limit of the run itself is not affected.")
    ap.add_argument("--skip-killed", action="store_true",
                    help="do not replay the blocks during which the container of the run was killed (memory "
                    "limit, crash, kill after the time limit). The state restore of the run re-executed only "
                    "the blocks that had ended normally, so such a block left no state there either.")
    a = ap.parse_args()
    global MEMORY, SKIP_KILLED
    MEMORY = a.memory
    SKIP_KILLED = a.skip_killed
    exp_dir = os.path.join(paths.RESULTS, a.exp)
    cells = sorted(os.path.dirname(p) for p in
                   glob.glob(os.path.join(exp_dir, "*", "*", "*", "run*", "T*", "cell_result.json")))
    if a.cells:
        want_c = {os.path.join(exp_dir, c.strip().strip("/")) for c in a.cells.split(",")}
        cells = [c for c in cells if c in want_c]
    if a.tasks:
        want = {f"T{int(x):02d}" for x in a.tasks.split(",")}
        cells = [c for c in cells if os.path.basename(c) in want]
    with ThreadPoolExecutor(a.jobs) as ex:
        for rec in ex.map(lambda c: eval_cell_safe(c, a.force), cells):
            fails = [c["name"] for c in rec.get("checks", []) if c["required"] and not c["passed"]]
            print(f"{rec['cell_id']:45s} {rec.get('status'):8s} correct={rec.get('analytically_correct')}"
                  f"{' SCORER_ERROR' if rec.get('scorer_error') else ''} "
                  f"replay_same_outputs={rec.get('replay', {}).get('same_outputs_as_original')} fails={fails}", flush=True)


if __name__ == "__main__":
    main()
