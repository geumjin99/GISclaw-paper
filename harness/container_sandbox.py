"""Host-side sandbox that executes code inside an isolated Docker container.

Public surface: execute, system_exec, add_prelude, inspect, get_full_code, reset, close,
work_dir, code_history, journal, events.

Inside the container the code sees only:
    /task/dataset       task inputs, read-only
    /task/pred_results  this cell's outputs (host: <work_dir>/pred_results)
    /task               scratch cwd (host: <work_dir>/_task_root)
No network, read-only root fs, tmpfs /tmp, memory/cpu/pids limits, non-root user.

The per-execution timeout is enforced here: on expiry the kernel gets SIGINT; if it does
not answer within the grace period the container is killed, a fresh one is started and
earlier blocks are re-run to restore the state.

Two callers use this class, and `journal_replay` selects how a session behaves for them:
    True   an agent run: after a restart every earlier block that ran to its end is re-run
           in order (failed ones included, they can leave partial state); a block that
           finishes within the grace period counts as finished; the message after a
           restart says what was restored.
    False  (default) a single script or a scoring replay: after a restart only the
           successful blocks are re-run; a block that exceeds the limit is a failure.
"""
import json
import os
import re
import select
import shutil
import subprocess
import time
import uuid

import paths
from sandbox import ExecutionResult, PythonSandbox

KERNEL = os.path.join(paths.HARNESS_DIR, "kernel.py")
SANDBOX_PY = os.path.join(paths.HARNESS_DIR, "sandbox.py")

NAME_PREFIX = "gisclaw"   # container names
LABEL = "gisclaw"         # container label; leftover containers are found by it

DEFAULTS = dict(image="gisclaw-exec:1.0", memory="6g", cpus="2", pids=512, tmpfs_size="2g",
                grace=15, start_timeout=120, blocked=None, label=LABEL, exp="adhoc",
                exec_log=None,          # path of a JSONL file recording every executed block, in order
                extra_outputs_dir=None, # host directory that receives a copy of /task/output
                journal_replay=False)   # see the module docstring


class SandboxError(RuntimeError):
    pass


class ContainerSandbox:
    def __init__(self, work_dir, timeout, dataset_dir, cell_id=None, **opts):
        self._ER = ExecutionResult
        # the import header PythonSandbox.get_full_code puts before the code blocks
        self._prefix = PythonSandbox.get_full_code(type("X", (), {"code_history": []})())
        self.work_dir = os.path.abspath(work_dir)
        self.timeout = timeout
        self.dataset_dir = os.path.realpath(dataset_dir)
        self.opts = {**DEFAULTS, **opts}
        self.cell_id = cell_id or uuid.uuid4().hex[:8]
        self.code_history = []
        self.journal = []         # every executed block in order: (code, success, outcome)
        self.system_preludes = [] # harness code re-run on every (re)start, never logged as agent code
        self.events = []          # timeouts / restarts, for the cell log
        self.fatal = False        # set when the sandbox could not be restored at all
        self._last_outcome = "ok"
        self.task_root = os.path.join(self.work_dir, "_task_root")
        self.pred_dir = os.path.join(self.work_dir, "pred_results")
        os.makedirs(self.task_root, exist_ok=True)
        os.makedirs(self.pred_dir, exist_ok=True)
        self.proc = None
        self.name = None
        self._start()

    # ------------------------------------------------------------------ container
    def _docker_cmd(self):
        o = self.opts
        uid, gid = os.getuid(), os.getgid()
        return [
            "docker", "run", "-i", "--name", self.name,
            "--label", f"{o['label']}=1", "--label", f"{o['label']}.exp={o['exp']}",
            "--label", f"cell={self.cell_id}",
            "--network", "none", "--read-only",
            "--tmpfs", f"/tmp:rw,size={o['tmpfs_size']},mode=1777",
            "--memory", o["memory"], "--memory-swap", o["memory"],
            "--cpus", str(o["cpus"]), "--pids-limit", str(o["pids"]),
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", f"{uid}:{gid}",
            "-e", "HOME=/tmp", "-e", "MPLCONFIGDIR=/opt/mplconfig", "-e", "PYTHONHASHSEED=0",
            # make every library see the container's CPU quota, not the host's core count;
            # otherwise thread and process pools differ from host to host
            *[x for var in ("OMP_NUM_THREADS", "PYTHON_CPU_COUNT", "NUMBA_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS", "LOKY_MAX_CPU_COUNT")
              for x in ("-e", f"{var}={o['cpus']}")],
            "-v", f"{self.task_root}:/task:rw",
            "-v", f"{self.dataset_dir}:/task/dataset:ro",
            "-v", f"{self.pred_dir}:/task/pred_results:rw",
            "-v", f"{SANDBOX_PY}:/opt/k/sandbox.py:ro",
            "-v", f"{KERNEL}:/opt/k/kernel.py:ro",
            "-w", "/task", o["image"], "python", "-u", "/opt/k/kernel.py",
        ]

    def _start(self):
        # docker names allow only [a-zA-Z0-9_.-]; ablation tiers carry "+" (e.g. H2+nodd)
        self.name = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{NAME_PREFIX}-{self.cell_id}-{uuid.uuid4().hex[:6]}")
        # stderr goes to a file: an unread pipe would fill up and block the kernel
        self._errlog = open(os.path.join(self.work_dir, "container_stderr.log"), "ab")
        self.proc = subprocess.Popen(self._docker_cmd(), stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self._errlog,
                                     text=True, bufsize=1)
        self._buf = b""
        self._next_id = 0
        r = self._call({"op": "init", "blocked": self.opts["blocked"]}, self.opts["start_timeout"])
        if r is None or not r.get("ok"):
            err = self._drain_stderr()
            self.close()
            raise SandboxError(f"container failed to start: {r} {err[-1500:]}")

    def _drain_stderr(self):
        try:
            with open(os.path.join(self.work_dir, "container_stderr.log"), "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4000))
                return f.read().decode(errors="replace")
        except Exception:
            return ""

    def _readline(self, deadline):
        fd = self.proc.stdout.fileno()
        while b"\n" not in self._buf:
            left = deadline - time.time()
            if left <= 0:
                return None
            r, _, _ = select.select([fd], [], [], min(left, 1.0))
            if not r:
                if self.proc.poll() is not None:
                    return None
                continue
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode("utf-8", errors="replace")

    def _send(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def _wait_reply(self, rid, deadline):
        """Read lines until the reply carrying request id `rid`; stray or stale lines
        (e.g. a late reply to an earlier, timed-out request) are discarded."""
        while True:
            line = self._readline(deadline)
            if line is None:
                return None
            try:
                obj = json.loads(line)
            except Exception:
                self.events.append({"t": time.time(), "event": "stray_protocol_line"})
                continue
            if isinstance(obj, dict) and obj.get("id") == rid:
                return obj
            self.events.append({"t": time.time(), "event": "stale_reply_discarded"})

    def _call(self, msg, timeout):
        """Send a request; return parsed reply, or None on timeout / dead container."""
        if self.proc is None or self.proc.poll() is not None:
            return None
        self._next_id += 1
        msg = dict(msg, id=self._next_id)
        self._last_id = self._next_id
        try:
            self._send(msg)
        except (BrokenPipeError, OSError):
            return None
        return self._wait_reply(self._next_id, time.time() + timeout)

    def _exit_info(self, wait_s=5.0):
        """If the container has stopped, return Docker's record of why; else None."""
        end = time.time() + wait_s
        while time.time() < end:
            r = subprocess.run(["docker", "inspect", "-f",
                                "{{.State.Running}} {{.State.ExitCode}} {{.State.OOMKilled}}", self.name],
                               capture_output=True, text=True)
            parts = r.stdout.split()
            if len(parts) == 3 and parts[0] == "false":
                return {"exit_code": int(parts[1]), "oom_killed": parts[2] == "true"}
            if r.returncode != 0:  # already removed
                return {"exit_code": None, "oom_killed": None}
            time.sleep(0.25)
        return None

    def _signal(self, sig):
        subprocess.run(["docker", "kill", "--signal", sig, self.name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _restart_and_replay(self):
        """Fresh container, then re-run the successful blocks to restore the variables.
        Best effort: a block that succeeded once can fail on replay for reasons that are
        not the agent's (e.g. os.mkdir without exist_ok, a file it has since moved). Such a
        block is skipped and recorded, and the restore continues with the next block.
        Returns the failed indices."""
        self.close()
        try:
            self._start()
        except Exception as e:
            self.events.append({"t": time.time(), "event": "restart_start_failed", "error": str(e)[:300]})
            self.fatal = True
            raise
        for code in self.system_preludes:
            self._call({"op": "exec", "code": code}, max(self.timeout, 60))
        if self.opts.get("journal_replay"):
            return self._replay_journal()
        failed = []
        for i, code in enumerate(self.code_history):
            r = self._call({"op": "exec", "code": code}, max(self.timeout, 60))
            if r is None or not r.get("ok") or not r["result"]["success"]:
                failed.append(i)
                if r is None and (self.proc is None or self.proc.poll() is not None):
                    # the container died during replay; nothing more can be restored
                    self.events.append({"t": time.time(), "event": "restart_start_failed",
                                        "error": f"container died replaying block {i}"})
                    self.fatal = True
                    raise SandboxError("container died during state replay")
        ev = {"t": time.time(), "event": "restarted", "replayed": len(self.code_history)}
        if failed:
            ev["replay_failed_blocks"] = failed
            self.events.append({"t": time.time(), "event": "replay_partial", "failed_blocks": failed,
                                "of": len(self.code_history)})
        self.events.append(ev)
        return failed

    def _replay_journal(self):
        """Re-run every earlier block in its original order, failed ones included
        (a failed block can leave partial state that later blocks depend on). Blocks that
        timed out or crashed the container are skipped. Returns the indices whose outcome
        differed from the original run."""
        differed = []
        for i, (code, ok, outcome) in enumerate(self.journal):
            if outcome != "ok":
                continue
            r = self._call({"op": "exec", "code": code}, max(self.timeout, 60))
            if r is None and (self.proc is None or self.proc.poll() is not None):
                self.events.append({"t": time.time(), "event": "restart_start_failed",
                                    "error": f"container died replaying block {i}"})
                self.fatal = True
                raise SandboxError("container died during state replay")
            got = bool(r and r.get("ok") and r["result"]["success"])
            if got != ok:
                differed.append(i)
        replayed = sum(1 for _, _, o in self.journal if o == "ok")
        ev = {"t": time.time(), "event": "restarted", "replayed": replayed}
        if differed:
            ev["replay_failed_blocks"] = differed
            self.events.append({"t": time.time(), "event": "replay_partial", "failed_blocks": differed,
                                "of": replayed})
        self.events.append(ev)
        self._replayed = replayed
        return differed

    # ------------------------------------------------------------------ system channel
    def system_exec(self, code, timeout=120):
        """Run harness code in the kernel (data profile, GIS guards, validation). Not part of
        the agent's code history, not written to exec_log, not replayed for scoring.
        Returns dict(stdout, stderr, success) or None if the kernel did not answer."""
        r = self._call({"op": "exec", "code": code}, timeout)
        if r is None or not r.get("ok"):
            return None
        res = r["result"]
        return {"stdout": res["stdout"], "stderr": res["stderr"], "success": res["success"]}

    def add_prelude(self, code, timeout=120):
        """Harness code that must be active in every kernel (re-run after a restart)."""
        self.system_preludes.append(code)
        return self.system_exec(code, timeout)

    def _scrub_symlinks(self):
        """Code in the container can create symlinks that point at host paths; anything the
        host later copies (outputs) must not follow them, so remove them."""
        for base in (self.pred_dir, self.task_root):
            for root, dirs, files in os.walk(base):
                for n in dirs + files:
                    p = os.path.join(root, n)
                    if os.path.islink(p):
                        try:
                            os.unlink(p)
                            self.events.append({"t": time.time(), "event": "symlink_removed", "path": os.path.relpath(p, self.work_dir)})
                        except OSError:
                            pass

    def _keep_extra_outputs(self):
        """Some task instructions ask for results under output/ while the system prompt says
        pred_results/; the work dir (and so output/) is deleted when the run ends. Mirror
        output/ into the cell directory after every block, so files saved there survive
        for scoring. Copy only; the agent never sees this."""
        dest = self.opts.get("extra_outputs_dir")
        src = os.path.join(self.task_root, "output")
        if not dest or not os.path.isdir(src):
            return
        try:
            shutil.copytree(src, os.path.join(dest, "output"), dirs_exist_ok=True, symlinks=False,
                            ignore_dangling_symlinks=True)
        except Exception as e:
            self.events.append({"t": time.time(), "event": "extra_outputs_copy_failed", "error": str(e)[:200]})

    # ------------------------------------------------------------------ public API
    def execute(self, code: str):
        t0 = time.time()
        r = None
        self._last_outcome = "exception"
        try:
            r = self._execute(code)
            return r
        finally:
            self._scrub_symlinks()
            self._keep_extra_outputs()
            if self.opts.get("exec_log"):
                # side record for faithful replay (failed blocks can leave partial state);
                # the agent never sees this file
                try:
                    with open(self.opts["exec_log"], "a") as f:
                        f.write(json.dumps({"t": round(t0, 3), "code": code,
                                            "success": bool(r and r.success),
                                            "timed_out": self._last_outcome.startswith("timeout"),
                                            "outcome": self._last_outcome,
                                            "elapsed": round(time.time() - t0, 2)}) + "\n")
                except OSError:
                    pass

    def _execute(self, code: str):
        t0 = time.time()
        self._last_outcome = "ok"
        r = self._call({"op": "exec", "code": code}, self.timeout)
        if r is not None and r.get("ok"):
            res = r["result"]
            if res.get("interrupted"):  # an interrupt that was not sent from here
                res["stderr"] += "\n\u23f0 Execution interrupted"
            er = self._ER(stdout=res["stdout"], stderr=res["stderr"], success=res["success"],
                          new_vars=res["new_vars"], execution_time=res["execution_time"])
            if er.success:
                self.code_history.append(code)
            self.journal.append((code, bool(er.success), "ok"))
            return er
        # timeout, or the container died (e.g. out-of-memory). Ask Docker which one.
        info = self._exit_info(wait_s=2.0 if r is None else 0.5)
        died = info is not None
        if died:
            how = "container_oom" if info.get("oom_killed") else f"container_exit_{info.get('exit_code')}"
        else:
            how = "timeout"
        if not died:
            self._signal("INT")
            late = self._wait_reply(self._last_id, time.time() + self.opts["grace"])
            if (late is not None and self.opts.get("journal_replay") and late.get("ok")
                    and not late["result"].get("interrupted")):
                # the block finished on its own within the grace period; report what it did
                res = late["result"]
                self.events.append({"t": time.time(), "event": "finished_in_grace"})
                er = self._ER(stdout=res["stdout"], stderr=res["stderr"], success=res["success"],
                              new_vars=res["new_vars"], execution_time=res["execution_time"])
                if er.success:
                    self.code_history.append(code)
                self.journal.append((code, bool(er.success), "ok"))
                return er
            if late is not None:
                if self.opts.get("journal_replay"):
                    # interrupted mid-way: its partial state is not re-created by a later restart
                    self.journal.append((code, False, "interrupted"))
                self.events.append({"t": time.time(), "event": "timeout_interrupted"})
                self._last_outcome = "timeout_interrupted"
                return self._ER(stdout="", success=False, execution_time=time.time() - t0,
                                stderr=f"\u23f0 Execution exceeded the {self.timeout}s limit and was interrupted. "
                                       f"Variables created before the interruption may be partial.")
        self.events.append({"t": time.time(), "event": how, **(info or {})})
        self._last_outcome = "timeout_killed" if how == "timeout" else ("oom" if how == "container_oom" else "crash")
        self.journal.append((code, False, self._last_outcome))
        failed = self._restart_and_replay()
        if self.opts.get("journal_replay"):
            n = getattr(self, "_replayed", 0)
            what = ("Execution exceeded the %ds limit and was terminated." % self.timeout if how == "timeout" else
                    "The execution environment ran out of memory." if how == "container_oom" else
                    "The execution environment crashed (exit code 137, which usually means it ran out of memory)."
                    if how == "container_exit_137" else f"The execution environment crashed ({how}).")
            restored = (f" A fresh environment was started and the {n} earlier code block(s) were re-run in order"
                        + (f"; {len(failed)} of them gave a different result than before, so some variables may"
                           f" differ or be missing." if failed else "; variables should be as before this block.")
                        + " Files this block wrote to disk may be incomplete.")
            return self._ER(stdout="", success=False, execution_time=time.time() - t0,
                            stderr=("\u23f0 " if how == "timeout" else "\u274c ") + what + restored)
        note = (f" {len(failed)} of {len(self.code_history)} earlier steps could not be re-run, so"
                f" variables they created may be missing." if failed else "")
        return self._ER(stdout="", success=False, execution_time=time.time() - t0,
                        stderr=(f"\u23f0 Execution exceeded the {self.timeout}s limit and was terminated."
                                if how == "timeout" else
                                ("\u274c The execution environment ran out of memory and was restarted."
                                 if how == "container_oom" else f"\u274c The execution environment crashed ({how})."))
                               + " The environment was restored to the state after the last successful step." + note)

    def _text(self, op, **kw):
        r = self._call({"op": op, **kw}, 60)
        if r is None or not r.get("ok"):
            return f"(unavailable: {r.get('error') if r else 'no reply'})"
        return r["text"]

    def inspect(self, var_name):
        return self._text("inspect", name=var_name)

    def _describe_var(self, var_name, detailed=False):
        return self._text("describe", name=var_name, detailed=detailed)

    def _list_user_vars(self):
        return self._text("list_vars")

    def get_variables_summary(self):
        return self._list_user_vars()

    def get_full_code(self):
        return self._prefix + "\n\n".join(self.code_history)

    def reset(self):
        self.code_history = []
        self.close()
        self._start()

    def close(self):
        if self.proc is not None:
            try:
                subprocess.run(["docker", "rm", "-f", self.name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=60)
                self.proc.wait(timeout=30)
            except Exception:
                pass
            for f in (self.proc.stdin, self.proc.stdout, getattr(self, "_errlog", None)):
                try:
                    f.close()
                except Exception:
                    pass
        self.proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
