#!/usr/bin/env python3
"""Run ONE cell: experiment x tier x architecture x model x run x task.

    python3 harness/run_cell.py --exp main --tier H2 --arch SA --model gptoss-120b --run 1 --task 1

Tiers: C0 (one-shot code generation), H0, H1, H2; an ablation is written as a suffix:
H2-g3 (H2 without component G3), H2+nodd / H2+nodk (without the task's Dataset Description /
Domain Knowledge), H2+hint (with the task's knowledge note). Architectures: SA (single
agent), DA (planner-worker dual agent).

The agent's code runs in an isolated container with the task inputs mounted read-only, under
the limits of config.yaml. Results go to <results>/<exp>/<tier>/<arch>/<model>/run<k>/T<id>/:
    cell_result.json  written last; its presence marks the cell as finished
    llm_calls.jsonl   every model call with the full reply
    rounds.jsonl      one line per agent round
    triggers.jsonl    harness components that fired
    prompt.json       system prompt and task message
    exec_log.jsonl    every executed code block in order (input of the scoring replay)
    <model>/          the deliverables and code.py (the successful blocks as one script)
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

sys.path.insert(0, paths.AGENT_DIR)
from container_sandbox import ContainerSandbox  # noqa: E402
import dual as agent_dual  # noqa: E402
import loop as agent_loop  # noqa: E402
import oneshot as agent_oneshot  # noqa: E402
from llm import ChatClient  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

CONFIG = os.path.join(paths.HARNESS_DIR, "config.yaml")


def cell_paths(exp, tier, arch, model, run, task):
    """(cell id, cell directory)."""
    cell_id = f"{exp}.{tier}.{arch}.{model}.r{run}.T{task:02d}"
    return cell_id, os.path.join(paths.RESULTS, exp, tier, arch, model, f"run{run}", f"T{task:02d}")


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def quick_fingerprint(d):
    """File list + sizes + mtimes. Inputs are read-only, so any change here is a red flag;
    the scheduler also does a full SHA-256 check before and after a batch."""
    out = []
    for root, _, files in os.walk(d):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out.append((os.path.relpath(p, d), st.st_size, st.st_mtime_ns))
    return hashlib.sha256(json.dumps(sorted(out)).encode()).hexdigest()


def is_heavy(data_dir, cfg):
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(data_dir) for f in fs)
    return total > cfg["container"]["heavy_input_mb"] * 1024 * 1024


def image_id(image):
    try:
        return subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", image],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def bind_to_parent():
    """A cell must never outlive its scheduler: an orphaned cell would keep running (and its
    sandbox would restart killed containers) while a new scheduler runs the same cell in the
    same directory. Ask the kernel to send SIGTERM when the parent dies, and turn SIGTERM
    into SystemExit so the sandbox is closed in `finally`."""
    def _term(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGHUP, _term)
    try:
        import ctypes
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass
    if os.getppid() == 1:  # parent already gone before prctl took effect
        raise SystemExit("scheduler is gone; not starting an orphaned cell")


def load_task(csv_path, task_id):
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["id"]) == task_id:
                return {"instruction": row.get("Instruction", "") or "",
                        "dataset_description": row.get("Dataset Description", "") or "",
                        "domain_knowledge": row.get("Domain Knowledge", "") or ""}
    raise SystemExit(f"task {task_id} not in {csv_path}")


def tier_config(tier):
    """C0 (one-shot baseline, agent/oneshot.py) or a loop tier (loop.tier_config)."""
    return agent_oneshot.tier_config() if tier == agent_oneshot.TIER else agent_loop.tier_config(tier)


def make_client(mcfg, record_path):
    if mcfg["provider"] == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit("ANTHROPIC_API_KEY is not set")
        from claude_client import ClaudeChat
        return ClaudeChat(mcfg["model_name"], api_key=key, request=mcfg.get("request"), record_path=record_path)
    return ChatClient(mcfg["base_url"], mcfg["model_name"], api_key="EMPTY",
                      request=mcfg.get("request"), record_path=record_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment name (first level of the results tree)")
    ap.add_argument("--tier", required=True, help="C0 | H0 | H1 | H2, optionally with ablations, e.g. H2-g1, H2+nodd")
    ap.add_argument("--arch", default="SA", choices=["SA", "DA"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--task-table", default=paths.TASKS_CSV, help="CSV with the task texts")
    ap.add_argument("--force", action="store_true", help="run again even if the cell is finished")
    a = ap.parse_args()
    bind_to_parent()
    cfg = load_config(a.config)
    agent_cfg = cfg.get("agent", {})
    tcfg = tier_config(a.tier)
    oneshot = a.tier == agent_oneshot.TIER
    if oneshot and a.arch != "SA":
        raise SystemExit("tier C0 is single-call: --arch SA only")
    cell_id, cell_dir = cell_paths(a.exp, a.tier, a.arch, a.model, a.run, a.task)
    marker = os.path.join(cell_dir, "cell_result.json")
    if os.path.exists(marker) and not a.force:
        print(f"[skip] {cell_id} already finished")
        return 0
    if os.path.exists(cell_dir):
        shutil.rmtree(cell_dir)
    os.makedirs(cell_dir)
    data_dir = os.path.join(paths.INPUTS, str(a.task), "dataset")
    if not os.path.isdir(data_dir):
        raise SystemExit(f"missing inputs: {data_dir}")
    task = load_task(a.task_table, a.task)
    fp_before = quick_fingerprint(data_dir)

    work_dir = os.path.join(cell_dir, "work")
    os.makedirs(work_dir)
    mem = cfg["container"]["heavy_memory"] if is_heavy(data_dir, cfg) else cfg["container"]["memory"]
    if tcfg.get("hint"):
        note = os.path.join(paths.NOTES, f"T{a.task:02d}.md")
        if not os.path.exists(note):
            raise SystemExit(f"tier +hint: no knowledge note {note}")
        with open(note, encoding="utf-8") as f:
            task["hint"] = f.read().strip()
    mcfg = cfg["models"][a.model]
    llm = make_client(mcfg, os.path.join(cell_dir, "llm_calls.jsonl"))
    sb = None
    t0 = time.time()
    status, err, stats = "done", None, {}
    try:
        sb = ContainerSandbox(work_dir, cfg["step_timeout_s"], dataset_dir=data_dir, cell_id=cell_id,
                              image=cfg["image"], memory=mem, cpus=cfg["container"]["cpus"],
                              pids=cfg["container"]["pids"], tmpfs_size=cfg["container"]["tmpfs_size"],
                              blocked=cfg.get("blocked_packages"), exp=a.exp, journal_replay=True,
                              exec_log=os.path.join(cell_dir, "exec_log.jsonl"),
                              extra_outputs_dir=os.path.join(cell_dir, "extra_outputs"))
        cls = agent_oneshot.OneShot if oneshot else (agent_dual.DualAgent if a.arch == "DA" else agent_loop.Agent)
        ag = cls(llm, sb, data_dir, cell_dir, tier=a.tier,
                 max_rounds=cfg["max_rounds"], budget_s=cfg["task_wall_s"],
                 # a model entry can set its own history budget
                 history_tokens=mcfg.get("history_tokens", agent_cfg.get("history_tokens", 60000)),
                 max_format_errors=agent_cfg.get("max_format_errors", 15))
        stats = ag.run(task)
        if str(stats.get("end_reason", "")).startswith("llm_error"):
            status, err = "llm_error", stats["end_reason"]
    except BaseException:
        status, err = "error", traceback.format_exc()[-4000:]
    finally:
        if sb is not None:
            sb.close()
    elapsed = time.time() - t0
    events = sb.events if sb is not None else []
    if status in ("done", "error") and any(ev["event"] == "restart_start_failed" for ev in events):
        status, err = "sandbox_error", f"sandbox could not be restored; {err or ''}"[:4000]
    pred = os.path.join(work_dir, "pred_results")
    outputs = sorted(f for f in os.listdir(pred) if not f.startswith(".")) if os.path.isdir(pred) else []
    # the deliverables and code.py go to <cell>/<model>/, where the scoring replay reads them;
    # work/ is removed below
    out_dir = os.path.join(cell_dir, a.model)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(pred):
        shutil.copytree(pred, out_dir, symlinks=False, ignore_dangling_symlinks=True, dirs_exist_ok=True)
    req = agent_loop.required_outputs(task["instruction"])
    result = {
        "cell_id": cell_id, "experiment": a.exp, "tier": a.tier, "arch": a.arch, "model": a.model,
        "run": a.run, "task": a.task, "status": status, "error": err, "elapsed_s": round(elapsed, 1),
        "protocol": PROTOCOL, "tier_config": tcfg,
        "outputs": outputs, "required_outputs": req,
        "required_present": (all(r in outputs for r in req) if req else None),
        "emergency_files": [f for f in outputs if f.startswith("emergency_")],
        "deliverable_produced": bool(stats.get("finished") and any(not f.startswith("emergency_") for f in outputs)),
        "rounds": stats.get("rounds"), "self_corrections": stats.get("self_corrections"),
        "agent_stats": stats, "llm_stats": getattr(llm, "stats", {}),
        "llm_time_s": round(getattr(llm, "llm_time", 0.0), 1),
        "agent_time_s": stats.get("agent_time_s", round(elapsed - getattr(llm, "llm_time", 0.0), 1)),
        "hit_deadline": stats.get("end_reason") == "budget",
        "sandbox_events": events, "inputs_intact": fp_before == quick_fingerprint(data_dir),
        "container_memory": mem, "image_id": image_id(cfg["image"]),
        "config": {k: cfg.get(k) for k in ("step_timeout_s", "task_wall_s", "max_rounds", "blocked_packages",
                                           "image", "agent")},
        "model_config": mcfg,
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if oneshot:
        result["oneshot_no_code"] = bool(stats.get("oneshot_no_code"))
        # C0 runs its one script with the whole task budget (agent/oneshot.py); record that as the
        # cell's execution limit, which the scoring replay scales like the limit of an agent block
        result["config"]["step_timeout_s"] = (stats.get("oneshot") or {}).get("timeout_s", int(cfg["task_wall_s"]))
        result["config"]["agent_step_timeout_s"] = cfg.get("step_timeout_s")
    try:
        full = sb.get_full_code() if sb is not None else ""
        with open(os.path.join(out_dir, "code.py"), "w") as f:
            f.write(full)
    except Exception:
        pass
    shutil.rmtree(work_dir, ignore_errors=True)
    tmp = marker + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=1, ensure_ascii=False, default=str)
    os.replace(tmp, marker)
    print(f"[cell] {cell_id} status={status} end={stats.get('end_reason')} rounds={stats.get('rounds')} "
          f"fmt_err={stats.get('format_errors')} outputs={len(outputs)} {elapsed:.0f}s", flush=True)
    return 0 if status == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
