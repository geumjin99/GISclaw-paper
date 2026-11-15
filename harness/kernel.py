"""Persistent Python kernel that runs INSIDE the gisclaw-exec container.

It wraps PythonSandbox (sandbox.py, mounted read-only next to this file at /opt/k).

Protocol: one JSON object per line on stdin, one JSON reply per line on the protocol fd
(a duplicate of the initial stdout). Real fd 1 is redirected to fd 2 so stray C-level prints cannot
corrupt the protocol.
    {"op": "init", "blocked": {...} | null}
    {"op": "exec", "code": "..."}          -> {"ok":1, "result": {stdout, stderr, success, new_vars, execution_time}}
    {"op": "inspect", "name": "x"}         -> {"ok":1, "text": "..."}
    {"op": "describe", "name": "x", "detailed": bool}
    {"op": "list_vars"}
    {"op": "ping"}
SIGINT during exec raises KeyboardInterrupt, which is reported as a failed execution.
"""
import json
import os
import sys
import traceback

import fcntl  # noqa: E402
proto = os.fdopen(fcntl.fcntl(1, fcntl.F_DUPFD_CLOEXEC, 200), "w", buffering=1)
os.dup2(2, 1)
sys.stdout = os.fdopen(1, "w", buffering=1)
# Commands arrive on a private copy of stdin; user code gets /dev/null, so exit()/quit()
# (which close sys.stdin) or input() cannot disturb the command channel.
cmd_in = os.fdopen(fcntl.fcntl(0, fcntl.F_DUPFD_CLOEXEC, 201), "r")
devnull = os.open(os.devnull, os.O_RDONLY)
os.dup2(devnull, 0)
sys.stdin = open(os.devnull)

sys.path.insert(0, "/opt/k")
from sandbox import PythonSandbox  # noqa: E402

import signal  # noqa: E402

sb = None
_busy = False


def _on_sigint(signum, frame):
    # Only interrupt user code; an INT that arrives while idle is ignored so it cannot
    # kill the kernel between requests.
    if _busy:
        raise KeyboardInterrupt


signal.signal(signal.SIGINT, _on_sigint)


_rid = None


def reply(obj):
    obj["id"] = _rid
    proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
    proto.flush()


def do_exec(code):
    global _busy
    _busy = True
    try:
        r = sb.execute(code)
        return {"stdout": r.stdout, "stderr": r.stderr, "success": r.success,
                "new_vars": r.new_vars, "execution_time": r.execution_time}
    except KeyboardInterrupt:
        os.chdir(sb.work_dir)
        return {"stdout": "", "stderr": "KeyboardInterrupt", "success": False,
                "new_vars": {}, "execution_time": 0.0, "interrupted": True}
    except BaseException:  # SystemExit etc. must not kill the kernel
        os.chdir(sb.work_dir)
        return {"stdout": "", "stderr": traceback.format_exc()[-3000:], "success": False,
                "new_vars": {}, "execution_time": 0.0}
    finally:
        _busy = False


for line in cmd_in:
    try:
        msg = json.loads(line)
    except Exception:
        reply({"ok": 0, "error": "bad json"})
        continue
    op = msg.get("op")
    _rid = msg.get("id")
    try:
        if op == "init":
            if msg.get("blocked") is not None:
                PythonSandbox.BLOCKED_PACKAGES = msg["blocked"]
            sb = PythonSandbox(work_dir="/task", timeout=10 ** 9)  # timeout enforced by host
            reply({"ok": 1, "pid": os.getpid()})
        elif op == "exec":
            reply({"ok": 1, "result": do_exec(msg["code"])})
        elif op == "inspect":
            reply({"ok": 1, "text": sb.inspect(msg["name"])})
        elif op == "describe":
            reply({"ok": 1, "text": sb._describe_var(msg["name"], detailed=bool(msg.get("detailed")))})
        elif op == "list_vars":
            reply({"ok": 1, "text": sb._list_user_vars()})
        elif op == "ping":
            reply({"ok": 1})
        else:
            reply({"ok": 0, "error": f"unknown op {op}"})
    except KeyboardInterrupt:
        reply({"ok": 0, "error": "interrupted"})
    except BaseException:
        reply({"ok": 0, "error": traceback.format_exc()[-2000:]})
