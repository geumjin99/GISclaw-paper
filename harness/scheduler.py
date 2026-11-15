#!/usr/bin/env python3
"""Run many cells in parallel, safely and resumably.

    python3 harness/scheduler.py --exp main --models gptoss-120b,qwen38-27b \
        --archs SA,DA --tiers H0,H1,H2 --runs 1,2,3 --tasks 1-50

Each cell is a separate `run_cell.py` process in its own session; the scheduler
  * checks the inputs (full SHA-256) before and after the batch,
  * removes containers left over from an earlier crash,
  * limits concurrency overall and per model provider,
  * kills a cell that exceeds task_hard_wall_s + grace (process group + its containers) and
    records it as `wall_timeout`,
  * skips cells that already have cell_result.json (resume), unless --retry-errors and
    the cell did not end as `done`,
  * writes <results>/<exp>/progress.json and one log per cell under <results>/<exp>/logs/.
"""
import argparse
import collections
import contextlib
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402
from container_sandbox import LABEL  # noqa: E402
from run_cell import CONFIG, cell_paths, is_heavy, load_config  # noqa: E402

HARD_GRACE_S = 900              # beyond the hard wall: time for the last step to end
MIN_FREE_DISK = 20 * 1024 ** 3  # new cells wait while the results disk has less free space


def parse_list(s, cast=str):
    out = []
    for part in s.split(","):
        part = part.strip()
        if cast is int and "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(cast(part))
    return out


def docker_kill_label(label):
    ids = subprocess.run(["docker", "ps", "-q", "--filter", f"label={label}"],
                         capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "kill", *ids], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return len(ids)


def live_processes(script, exp):
    """PIDs of running `<script> ... --exp <exp>` processes other than this one."""
    mine, p = set(), os.getpid()
    while p > 1:  # this process and its ancestors (e.g. a `timeout` or `nohup` wrapper)
        mine.add(p)
        try:
            p = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in mine:
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        argv = [x.decode(errors="replace") for x in argv]
        if not argv or not os.path.basename(argv[0]).startswith("python"):
            continue
        if any(x.endswith(script) for x in argv) and "--exp" in argv \
                and argv.index("--exp") + 1 < len(argv) and argv[argv.index("--exp") + 1] == exp:
            out.append(int(pid))
    return out


def verify_inputs():
    r = subprocess.run([sys.executable, os.path.join(paths.HARNESS_DIR, "prepare_inputs.py"), "--verify"],
                       capture_output=True, text=True)
    print(r.stdout.strip())
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--archs", default="SA,DA")
    ap.add_argument("--tiers", default="H0,H1,H2", help="e.g. C0,H0,H1,H2,H2-g1,H2+nodd")
    ap.add_argument("--runs", default="1")
    ap.add_argument("--tasks", default="1-50")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--task-table", default=paths.TASKS_CSV, help="CSV with the task texts")
    ap.add_argument("--retry-errors", action="store_true",
                    help="re-run cells that ended in any status other than done")
    ap.add_argument("--total", type=int, help="override overall concurrency")
    ap.add_argument("--dry", action="store_true", help="list cells and exit")
    a = ap.parse_args()
    cfg = load_config(a.config)
    data_root = paths.INPUTS

    tasks = [t for t in parse_list(a.tasks, int) if os.path.isdir(os.path.join(data_root, str(t), "dataset"))]
    cells = [dict(model=m, arch=ar, tier=ti, run=r, task=t)
             for r in parse_list(a.runs, int) for t in tasks
             for m in parse_list(a.models) for ar in parse_list(a.archs) for ti in parse_list(a.tiers)]
    todo = []
    for c in cells:
        cid, cdir = cell_paths(a.exp, c["tier"], c["arch"], c["model"], c["run"], c["task"])
        c["id"], c["dir"] = cid, cdir
        mk = os.path.join(cdir, "cell_result.json")
        if os.path.exists(mk):
            st = json.load(open(mk)).get("status")
            if not (a.retry_errors and st != "done"):
                continue
        todo.append(c)
    print(f"{len(cells)} cells, {len(cells) - len(todo)} already finished, {len(todo)} to run")
    if a.dry:
        for c in todo[:20]:
            print(" ", c["id"])
        return 0
    if not todo:
        return 0

    stray = live_processes("run_cell.py", a.exp) + live_processes("scheduler.py", a.exp)
    if stray:
        print(f"refusing to start: processes of exp {a.exp} still running: {stray}")
        return 4
    if not verify_inputs():
        print("inputs failed verification; refusing to start")
        return 2
    n = docker_kill_label(f"{LABEL}.exp={a.exp}")  # only this experiment's leftovers
    if n:
        print(f"removed {n} leftover containers")

    exp_dir = os.path.join(paths.RESULTS, a.exp)
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    conc = cfg["concurrency"]
    n_total = a.total or conc["total"]
    # memory guard: the concurrent cells must fit into the host's RAM
    try:
        host_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        # container.typical_gb is the memory a typical cell uses; the per-container limit
        # (container.memory) still caps each cell, and heavy tasks have their own gate
        mem_gb = float(cfg["container"].get("typical_gb", str(cfg["container"]["memory"]).rstrip("gG")))
        fit = max(1, int(host_gb * 0.85 // mem_gb))
        if n_total > fit:
            print(f"concurrency {n_total} x {mem_gb:g} GB exceeds 85% of host RAM ({host_gb:.0f} GB): using {fit}")
            n_total = fit
    except Exception:
        pass
    total = threading.Semaphore(n_total)
    prov = {p: threading.Semaphore(v) for p, v in conc.items() if p not in ("total", "heavy")}
    # absolute safety cap; the experiment's budget (agent time) is enforced inside the cell
    hard_s = cfg.get("task_hard_wall_s", cfg["task_wall_s"]) + HARD_GRACE_S
    lock = threading.Lock()
    counts = collections.Counter()
    started = time.time()
    stop = threading.Event()
    running = {}

    def write_progress():
        json.dump({"exp": a.exp, "total": len(todo), "counts": dict(counts),
                   "elapsed_s": round(time.time() - started),
                   "updated": datetime.datetime.now().isoformat(timespec="seconds")},
                  open(os.path.join(exp_dir, "progress.json"), "w"), indent=1)

    heavy = threading.Semaphore(conc.get("heavy", 1))

    def run(c):
        p = cfg["models"][c["model"]]["provider"]
        hv = heavy if is_heavy(os.path.join(data_root, str(c["task"]), "dataset"), cfg) else contextlib.nullcontext()
        with prov[p], hv, total:
            if stop.is_set():
                return
            cmd = [sys.executable, "-W", "ignore", os.path.join(paths.HARNESS_DIR, "run_cell.py"), "--exp", a.exp,
                   "--tier", c["tier"], "--arch", c["arch"], "--model", c["model"],
                   "--run", str(c["run"]), "--task", str(c["task"]), "--config", a.config,
                   "--task-table", a.task_table, "--force"]
            log = open(os.path.join(log_dir, c["id"] + ".log"), "w")
            t0 = time.time()
            while shutil.disk_usage(exp_dir).free < MIN_FREE_DISK and not stop.is_set():
                print("low disk space: waiting", flush=True)
                time.sleep(60)
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=paths.ROOT,
                                    start_new_session=True)
            with lock:
                running[c["id"]] = proc
            try:
                proc.wait(timeout=hard_s)
                status = None
            except subprocess.TimeoutExpired:
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        proc.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        pass
                status = "wall_timeout"
            docker_kill_label(f"cell={c['id']}")
            with lock:
                running.pop(c["id"], None)
            log.close()
            mk = os.path.join(c["dir"], "cell_result.json")
            if status == "wall_timeout" or not os.path.exists(mk):
                os.makedirs(c["dir"], exist_ok=True)
                json.dump({"cell_id": c["id"], **{k: c[k] for k in ("model", "arch", "tier", "run", "task")},
                           "exp": a.exp, "status": status or "crashed", "returncode": proc.returncode,
                           "elapsed_s": round(time.time() - t0, 1)}, open(mk, "w"), indent=1)
            st = json.load(open(mk)).get("status")
            with lock:
                counts[st] += 1
                done = sum(counts.values())
                write_progress()
                print(f"[{done}/{len(todo)}] {c['id']} {st} {time.time() - t0:.0f}s", flush=True)

    def shutdown(signum, frame):
        # a scheduler started in the background ignores SIGINT, and `kill` sends SIGTERM;
        # either way the running cells are stopped as well
        stop.set()
        print(f"signal {signum}: stopping running cells", flush=True)
        with lock:
            procs = list(running.values())
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pr in procs:
                try:
                    os.killpg(pr.pid, sig)
                except ProcessLookupError:
                    pass
            deadline = time.time() + 20
            while time.time() < deadline and any(pr.poll() is None for pr in procs):
                time.sleep(0.5)
        docker_kill_label(f"{LABEL}.exp={a.exp}")
        print(f"stopped; finished so far: {dict(counts)}", flush=True)
        os._exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, shutdown)

    threads = [threading.Thread(target=run, args=(c,), daemon=True) for c in todo]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        stop.set()
        print("interrupted: waiting for running cells to be killed")
        with lock:
            for pr in list(running.values()):
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(pr.pid, sig)
                    except ProcessLookupError:
                        pass
                    time.sleep(1)
        docker_kill_label(f"{LABEL}.exp={a.exp}")
    ok = verify_inputs()
    print(f"done in {time.time() - started:.0f}s: {dict(counts)}; inputs intact after batch: {ok}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
