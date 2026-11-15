"""Tier C0: one-shot code generation baseline (no agent loop).

Same task message as H0 (Instruction + Dataset Description + Domain Knowledge + file list; no
data profile, no GIS layer, no knowledge note) and the same environment description (packages,
dataset/ read-only, pred_results/). The system prompt asks for ONE complete, directly runnable
Python script. One model call (no tools are sent) -> extract the script -> run it ONCE in the
same container sandbox -> end. No error feedback, no retry, no tools. A reply without a code
block is a failed cell with no outputs (oneshot_no_code).

Time limit: the agent tiers get up to task_wall_s of agent time per task, spread over several
executions of at most step_timeout_s each; the one-shot script does all the work in one
execution, so it gets the whole task budget: the sandbox's execution limit is set to budget_s
(= task_wall_s) for this one block. run_cell records this limit as the cell's
config.step_timeout_s, so the scoring replay scales it by the same factor as the limit of an
agent block.

Empty reply: as in every other tier, a completely empty reply (no text at all) is asked for
once more (counted in empty_retries); a reply with text but no code is final.

Run header: the executed block is RUN_HEADER + the model's script verbatim. The header makes
the script run as `python solution.py` in the task directory would: the kernel's namespace has
no __name__ (an `if __name__ == "__main__":` guard would otherwise skip main()) and no
__file__. It is part of the block written to exec_log.jsonl, so the scoring replay runs
exactly the same code.
"""
import json
import os
import re
import time

from llm import ContextOverflow, LLMFailure
from observe import format_exec
from prompts import NOT_AVAILABLE, PACKAGES, task_message
from protocol import PROTOCOL
from tools import list_files

TIER = "C0"
SHOW_WARNINGS = "import warnings as _w\n_w.resetwarnings()\n_w.simplefilter('default')\ndel _w\n"  # same as loop.SHOW_WARNINGS
RUN_HEADER = '__name__ = "__main__"; __file__ = "/task/solution.py"  # run as a script in the task directory\n'

SYSTEM = """You are an assistant that completes data analysis tasks by writing Python code.

## Environment
- The Python script in your reply is run once, as written, in the task directory. There is no internet access. You will not see its output, and there is no second attempt.
- Input data is in dataset/ (read-only). Save every deliverable in pred_results/.
- Available packages: {packages}. Not available: {not_available}.
- Figures: use matplotlib with plt.savefig(...); plt.show() does nothing.

## Answer format
Reply with one complete, directly runnable Python script in a ```python code block. The script must do the whole task, from reading the inputs to saving every deliverable in pred_results/. If your reply contains more than one ```python block, only the last one is run."""

PY_TAGS = ("python", "py", "python3")
_FENCE = re.compile(r"^[ \t]*```[ \t]*([\w+.-]*)[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.M | re.S)


def tier_config():
    return {"tools": [], "generic": False, "oneshot": True, "no_dd": False, "no_dk": False,
            **{g: False for g in ("g1", "g2", "g3", "g4", "g5", "g6", "g7")}}


def system_prompt():
    return SYSTEM.format(packages=PACKAGES, not_available=NOT_AVAILABLE)


def extract_code(text):
    """Code extraction rule for C0:
    1. Only complete fenced blocks count: an opening line ```<tag> and a closing line ```
       (a block cut off by the token limit, without its closing fence, is not code).
    2. The LAST block whose tag is python / py / python3 (case-insensitive) is the script.
    3. If there is no such block: the LONGEST fenced block with any or no tag.
    4. Blocks that are empty after stripping are ignored. No block -> (None, None, n).
    Returns (code, rule, n_fenced_blocks); rule is "last_python" or "longest_fence"."""
    blocks = [(m.group(1).lower(), m.group(2)) for m in _FENCE.finditer(text or "")]
    blocks = [(t, c) for t, c in blocks if c.strip()]
    py = [c for t, c in blocks if t in PY_TAGS]
    if py:
        return py[-1], "last_python", len(blocks)
    if blocks:
        return max((c for _, c in blocks), key=len), "longest_fence", len(blocks)
    return None, None, len(blocks)


class OneShot:
    """Same constructor as loop.Agent; run() returns the stats dict."""

    def __init__(self, llm, sandbox, dataset_dir, cell_dir, tier=TIER, budget_s=1200,
                 verbose=True, **_):
        if tier != TIER:
            raise ValueError(f"OneShot runs tier {TIER} only, not {tier}")
        self.llm, self.sb, self.dataset_dir, self.cell_dir = llm, sandbox, dataset_dir, cell_dir
        self.tier, self.cfg, self.verbose = tier, tier_config(), verbose
        self.script_timeout_s = int(budget_s)  # the whole task budget for the one execution
        self.stats = {"rounds": 0, "format_errors": 0, "format_error_kinds": {}, "self_corrections": 0,
                      "exec_blocks": 0, "exec_failed": 0, "finish_rejections": 0, "dedup_blocked": 0,
                      "triggers": {}, "checks_run": {}, "end_reason": None, "emergency_saved": [],
                      "empty_retries": 0, "history_truncations": 0, "oneshot_no_code": False,
                      "oneshot": {"code_rule": None, "n_fenced_blocks": 0, "code_chars": 0,
                                  "finish_reason": None, "timeout_s": int(budget_s)}}

    def _log(self, name, rec):
        with open(os.path.join(self.cell_dir, name), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def run(self, task):
        t0, llm0 = time.time(), self.llm.llm_time
        self.sb.add_prelude(SHOW_WARNINGS)
        shown = {k: task.get(k, "") for k in ("instruction", "dataset_description", "domain_knowledge")}
        self.sysmsg, self.user = system_prompt(), task_message(shown, list_files(self.dataset_dir), None)
        with open(os.path.join(self.cell_dir, "prompt.json"), "w") as f:
            json.dump({"tier": self.tier, "protocol": PROTOCOL, "config": self.cfg,
                       "system": self.sysmsg, "task": self.user}, f, ensure_ascii=False, indent=1)
        msgs = [{"role": "system", "content": self.sysmsg}, {"role": "user", "content": self.user}]
        st, finished = self.stats, False
        try:
            reply = self.llm.chat(msgs, tools=None, kind="oneshot")
            if not reply.content.strip() and not reply.tool_calls:
                st["empty_retries"] += 1
                reply = self.llm.chat(msgs, tools=None, kind="oneshot_retry_empty")
        except ContextOverflow:
            st["end_reason"] = "context_overflow"
            reply = None
        except LLMFailure as e:
            st["end_reason"] = f"llm_error: {str(e)[:200]}"
            reply = None
        if reply is not None:
            st["rounds"] = 1
            st["oneshot"]["finish_reason"] = reply.finish_reason
            code, rule, nb = extract_code(reply.content)
            st["oneshot"].update(code_rule=rule, n_fenced_blocks=nb, code_chars=len(code or ""))
            if code is None:
                st["oneshot_no_code"] = True
                st["end_reason"] = "no_code"
                obs, ok = "Not run: the reply contains no code block.", None
            else:
                self.sb.timeout = self.script_timeout_s
                res = self.sb.execute(RUN_HEADER + code)
                st["exec_blocks"], st["exec_failed"] = 1, int(not res.success)
                obs, ok = format_exec(res), bool(res.success)
                st["end_reason"], finished = "oneshot", True
            self._log("rounds.jsonl", {"n": 1, "kind": "oneshot", "reply": reply.content,
                                       "action": "execute" if code is not None else None,
                                       "code_rule": rule, "observation": obs, "success": ok})
            if self.verbose:
                print(f"  C0 {rule or 'no code'} -> {obs.splitlines()[0][:100]}", flush=True)
        st["finished"] = finished
        st["agent_time_s"] = round(time.time() - t0 - (self.llm.llm_time - llm0), 1)
        return st
