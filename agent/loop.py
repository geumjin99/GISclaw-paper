"""Single-agent ReAct loop.

One loop, three tiers as switch settings (TIERS); an ablation turns one switch off.
Base infrastructure (same in every tier): protocol parser, observation limits, token-budget
history, raw-reply logging, library warnings shown, restart notes that say what was restored.
"""
import hashlib
import json
import os
import re
import time

import gis_errors
import gis_runtime
from gis_profile import PROFILE_CODE
from gis_validate import VALIDATE_CODE
from history import History
from llm import ContextOverflow, LLMFailure
from observe import cap, last_error_line
from prompts import system_prompt, task_message
from protocol import PROTOCOL, FormatError, parse_reply, tool_schema
from tools import Toolkit, list_files

ALL_TOOLS = ["list_files", "load_vector", "load_raster", "inspect", "execute", "finish"]
GIS = ("g1", "g2", "g3", "g4", "g5", "g6", "g7")
TIERS = {
    "H0": dict(tools=["execute", "finish"], generic=False, **{g: False for g in GIS}),
    "H1": dict(tools=ALL_TOOLS, generic=True, **{g: False for g in GIS}),
    "H2": dict(tools=ALL_TOOLS, generic=True, **{g: True for g in GIS}),
}
PIN_ACTIONS = {"list_files", "load_vector", "load_raster", "inspect"}
SHOW_WARNINGS = "import warnings as _w\n_w.resetwarnings()\n_w.simplefilter('default')\ndel _w\n"
REQ_RX = re.compile(r"pred_results/([\w\-. ]+?\.(?:png|jpg|tiff|tif|geojson|csv|shp|gpkg|json|txt|html))\b")


def _raw(reply):
    """Provider-native assistant content to send back unchanged (Claude), or None."""
    r = getattr(reply, "raw", None)
    return r if r else None


def tier_config(tier):
    """'H2', 'H2-g3' (H2 with G3 switched off), 'H2+nodd' / 'H2+nodk' (without the human-written
    Dataset Description / Domain Knowledge; the system's file list is always given), 'H2+hint'
    (with the per-task knowledge note, task["hint"])."""
    tier, *inputs = tier.split("+")
    base, _, off = tier.partition("-")
    cfg = dict(TIERS[base])
    for o in filter(None, off.split("-")):
        if o not in GIS:
            raise ValueError(f"unknown component {o}")
        cfg[o] = False
    for i in inputs:
        if i not in ("nodd", "nodk", "hint"):
            raise ValueError(f"unknown input ablation {i}")
    cfg["no_dd"], cfg["no_dk"] = "nodd" in inputs, "nodk" in inputs
    if "hint" in inputs:
        cfg["hint"] = True
    if cfg["g6"]:  # G6 adds a tool
        cfg["tools"] = cfg["tools"][:-1] + ["lookup_api"] + cfg["tools"][-1:]
    return cfg


def required_outputs(instruction):
    return sorted(set(REQ_RX.findall(instruction or "")))


def _norm(code):
    lines = [l.strip() for l in (code or "").splitlines() if l.strip() and not l.strip().startswith("#")]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


EMERGENCY_CODE = r'''
def _gg_emergency(_names):
    import os, matplotlib.pyplot as _plt
    saved = []
    if not any(f.lower().endswith((".png", ".jpg")) for f in os.listdir("pred_results")):
        for i, n in enumerate(_plt.get_fignums()):
            f = _plt.figure(n)
            if f.axes:
                p = f"pred_results/emergency_plot_{i}.png"; f.savefig(p, dpi=150, bbox_inches="tight"); saved.append(p)
    import geopandas as _gpd, pandas as _pd
    g = globals()
    for n in reversed(_names):
        o = g.get(n)
        if isinstance(o, _gpd.GeoDataFrame) and len(o):
            o.to_file("pred_results/emergency_output.geojson", driver="GeoJSON"); saved.append(n); break
        if isinstance(o, _pd.DataFrame) and len(o):
            o.to_csv("pred_results/emergency_output.csv", index=False); saved.append(n); break
    return saved
print(_gg_emergency(__NAMES__))
del _gg_emergency
'''


class Agent:
    def __init__(self, llm, sandbox, dataset_dir, cell_dir, tier="H2", max_rounds=50,
                 budget_s=1200, history_tokens=60000, max_format_errors=15, verbose=True):
        self.llm, self.sb, self.dataset_dir, self.cell_dir = llm, sandbox, dataset_dir, cell_dir
        self.tier, self.cfg = tier, tier_config(tier)
        self.max_rounds, self.budget_s = max_rounds, budget_s
        self.history_tokens, self.max_format_errors = history_tokens, max_format_errors
        self.verbose = verbose
        self.tools = self.cfg["tools"]
        self.kit = Toolkit(sandbox, dataset_dir, self.tools)
        self.stats = {"rounds": 0, "format_errors": 0, "format_error_kinds": {}, "self_corrections": 0,
                      "exec_blocks": 0, "exec_failed": 0, "finish_rejections": 0, "dedup_blocked": 0,
                      "triggers": {}, "checks_run": {}, "end_reason": None, "emergency_saved": [],
                      "empty_retries": 0}

    # ------------------------------------------------------------------ logging
    def _log(self, name, rec):
        with open(os.path.join(self.cell_dir, name), "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def _trigger(self, comp, kind, detail, n):
        k = f"{comp}:{kind}"
        self.stats["triggers"][k] = self.stats["triggers"].get(k, 0) + 1
        self._log("triggers.jsonl", {"round": n, "comp": comp, "kind": kind, "detail": detail})

    def _say(self, s):
        if self.verbose:
            print(s, flush=True)

    # ------------------------------------------------------------------ run
    def setup(self, task):
        """Kernel preludes, data card, prompts. Shared by the single and the dual agent."""
        self.t0 = time.time()
        self.llm0 = self.llm.llm_time
        self.sb.add_prelude(SHOW_WARNINGS)
        pre = gis_runtime.prelude(self.cfg)
        if pre:
            r = self.sb.add_prelude(pre, timeout=300)
            if not (r and r["success"]):
                raise RuntimeError(f"GIS runtime prelude failed: {(r or {}).get('stderr', r)}")
        self.card = None
        if self.cfg["g1"]:
            r = self.sb.system_exec(PROFILE_CODE, timeout=300)
            self.card = (r or {}).get("stdout", "").strip() or "(data profile unavailable)"
            self._trigger("g1", "card", {"chars": len(self.card)}, 0)
        self.files = list_files(self.dataset_dir)
        self.sysmsg = system_prompt(self.tools, self.cfg["generic"], self.cfg["g5"])
        shown = dict(task)
        if self.cfg.get("no_dd"):
            shown["dataset_description"] = ""
        if self.cfg.get("no_dk"):
            shown["domain_knowledge"] = ""
        if self.cfg.get("hint") and not (task.get("hint") or "").strip():
            raise RuntimeError("tier +hint but the task has no hint")
        if not self.cfg.get("hint"):
            shown.pop("hint", None)
        self.user = task_message(shown, self.files, self.card)
        with open(os.path.join(self.cell_dir, "prompt.json"), "w") as f:
            json.dump({"tier": self.tier, "protocol": PROTOCOL, "config": self.cfg,
                       "system": self.sysmsg, "task": self.user}, f, ensure_ascii=False, indent=1)
        self.schema = tool_schema(self.tools)
        self.req = required_outputs(task.get("instruction"))
        self.seen, self.created, self.loaded, self.notes_sent = {}, [], set(), set()
        self.readonly = {}
        self.prev_failed = False
        self.n = 0

    def agent_time(self):
        return time.time() - self.t0 - (self.llm.llm_time - self.llm0)

    def run(self, task):
        self.setup(task)
        hist = History(self.sysmsg, self.user, self.history_tokens)
        finished, _ = self.react(hist, self.max_rounds, final=True)
        return self.wrapup(finished, [hist])

    def react(self, hist, limit, final=True, kind="agent"):
        """ReAct until finish, `limit` more rounds, the budget, or a model failure.
        final=False (a dual-agent step): finish ends the step without deliverable checks.
        Returns (finished, last finish summary)."""
        stop_at = self.n + limit
        while self.n < min(stop_at, self.max_rounds):
            if self.agent_time() > self.budget_s:
                self.stats["end_reason"] = "budget"
                return False, None
            try:
                reply = self.llm.chat(hist.messages(), tools=self.schema, kind=kind)
                if not reply.content.strip() and not reply.tool_calls:
                    self.stats["empty_retries"] += 1
                    reply = self.llm.chat(hist.messages(), tools=self.schema, kind=kind + "_retry_empty")
            except ContextOverflow:
                if hist.budget > 12000:
                    hist.budget = int(hist.budget * 0.6)
                    self._say(f"  [context overflow -> history budget {hist.budget}]")
                    continue
                self.stats["end_reason"] = "context_overflow"
                return False, None
            except LLMFailure as e:
                self.stats["end_reason"] = f"llm_error: {str(e)[:200]}"
                return False, None
            act = parse_reply(reply.content, reply.tool_calls, self.tools)
            if isinstance(act, FormatError):
                self.stats["format_errors"] += 1
                fk = self.stats["format_error_kinds"]
                fk[act.kind] = fk.get(act.kind, 0) + 1
                self._log("rounds.jsonl", {"n": self.n, "format_error": act.kind, "message": act.message,
                                           "reply": reply.content[:4000]})
                hist.add([{"role": "assistant", "content": reply.content or "(empty)", "_raw": _raw(reply)},
                          {"role": "user", "content": f"Format error: {act.message}"}], n=self.n)
                if self.stats["format_errors"] >= self.max_format_errors:
                    self.stats["end_reason"] = "format_errors"
                    return False, None
                continue
            self.n += 1
            n = self.n
            self.stats["rounds"] = n
            if act.name == "finish" and not final:
                obs, extra = "Step marked complete.", {"finished": True}
            else:
                obs, extra = self._act(act, n, self.req, self.seen, self.created, self.loaded, self.notes_sent)
            res = extra.get("res")
            if res is not None:
                self.stats["exec_blocks"] += 1
                if not res.success:
                    self.stats["exec_failed"] += 1
                if self.prev_failed and res.success:
                    self.stats["self_corrections"] += 1
                self.prev_failed = not res.success
            if act.note:
                obs = act.note + "\n" + obs
            finished = extra.get("finished", False)
            if self.cfg["generic"] and not finished:
                left = self.max_rounds - n
                if left in (5, 2):
                    obs += f"\n\n[System: {left} rounds left. Make sure every deliverable is saved in pred_results/.]"
                    self._trigger("h1", "rounds_left", {"left": left}, n)
                tleft = self.budget_s - self.agent_time()
                if tleft < 180 and "time" not in self.notes_sent:
                    self.notes_sent.add("time")
                    obs += "\n\n[System: less than 3 minutes of working time left. Save the deliverables now.]"
                    self._trigger("h1", "time_left", {"s": round(tleft)}, n)
            self._log("rounds.jsonl", {"n": n, "kind": kind, "thought": act.thought[:3000], "action": act.name,
                                       "args": act.args, "observation": obs[:8000],
                                       "success": (res.success if res is not None else None)})
            self._say(f"  R{n} {act.name} -> {obs.splitlines()[0][:100] if obs else ''}")
            hist.add(self._round_msgs(reply, act, obs), pinned=act.name in PIN_ACTIONS, n=n)
            if finished:
                if final:
                    self.stats["end_reason"] = "finish"
                return True, act.args.get("summary", "")
        if self.n >= self.max_rounds:
            self.stats["end_reason"] = self.stats["end_reason"] or "max_rounds"
        return False, None

    def wrapup(self, finished, hists):
        if not finished and self.cfg["generic"]:
            names = [c for c in self.created if c not in self.loaded]
            r = self.sb.system_exec(EMERGENCY_CODE.replace('__NAMES__', repr(names)), timeout=120)
            self.stats["emergency_saved"] = (r or {}).get("stdout", "").strip()
            self._trigger("h1", "emergency_save", {"saved": self.stats["emergency_saved"]}, self.n)
        self.stats["end_reason"] = self.stats["end_reason"] or ("finish" if finished else "max_rounds")
        if self.cfg["g7"]:
            self._write_provenance()
        self.stats["history_truncations"] = sum(h.truncations for h in hists)
        self.stats["finished"] = finished
        self.stats["agent_time_s"] = round(self.agent_time(), 1)
        return self.stats

    # ------------------------------------------------------------------ one action
    def _act(self, act, n, req, seen, created, loaded, notes_sent):
        name, args = act.name, act.args
        if name == "finish":
            return self._finish(n, req)
        if name == "execute" and self.cfg["generic"]:
            h = _norm(args.get("code"))
            if h in seen:
                k, ok, err = seen[h]
                self.stats["dedup_blocked"] += 1
                self._trigger("h1", "dedup", {"same_as": k, "prev_success": ok}, n)
                if ok:
                    return (f"Not run: this code is identical to round {k}, which already ran successfully. "
                            f"Its variables and files already exist; continue with the next step."), {}
                return (f"Not run: this code is identical to round {k}, which failed with: {err}\n"
                        f"Change the code before running it again."), {}
        obs, res, meta = self.kit.run(name, args)
        if res is None and self.cfg["generic"] and name in ("list_files", "inspect", "lookup_api"):
            # read-only call repeated with the same arguments and the same answer
            key = (name, json.dumps(args, sort_keys=True))
            prev = self.readonly.get(key)
            h = hashlib.sha256(obs.encode()).hexdigest()
            self.readonly[key] = (n, h)
            if prev and prev[1] == h:
                self.stats["dedup_blocked"] += 1
                self._trigger("h1", "repeat_readonly", {"tool": name, "same_as": prev[0]}, n)
                return (f"Same result as round {prev[0]}: nothing has changed since then. "
                        f"Continue with the next step of the analysis."), {}
        if res is not None:
            created.extend(k for k in (res.new_vars or {}) if k not in created)
            loaded.update(meta.get("loaded", []))
            if name == "execute" and self.cfg["generic"]:
                seen[_norm(args.get("code"))] = (n, res.success, last_error_line(res.stderr))
            gis_notes = []
            if any(self.cfg[g] for g in ("g2", "g3", "g7")):
                r = self.sb.system_exec(gis_runtime.post_code(self.cfg, meta.get("code", "")), timeout=120)
                for g in ("g2", "g3", "g7"):
                    if self.cfg[g]:
                        self.stats["checks_run"][g] = self.stats["checks_run"].get(g, 0) + 1
                try:
                    out = json.loads((r or {}).get("stdout", "").strip().splitlines()[-1])
                except Exception:
                    out = {"notes": [], "io": []}
                for nt in out.get("notes", []):
                    gis_notes.append(nt["msg"])
                    self._trigger(nt.get("comp", "g2"), nt["kind"], {"msg": nt["msg"][:300]}, n)
                if self.cfg["g7"]:
                    self._log("provenance.jsonl", {"round": n, "action": name, "success": res.success,
                                                   "code_sha": hashlib.sha256((meta.get("code") or "").encode()).hexdigest()[:12],
                                                   "io": out.get("io", []), "new_vars": sorted((res.new_vars or {}).keys())})
            if self.cfg["g6"] and not res.success:
                self.stats["checks_run"]["g6"] = self.stats["checks_run"].get("g6", 0) + 1
                r6 = self.sb.system_exec(gis_runtime.suggest_code((res.stderr or "")[-2000:]), timeout=180)
                sug = ((r6 or {}).get("stdout") or "").strip()
                if sug:
                    gis_notes.append(sug)
                    self._trigger("g6", "suggest", {"text": sug[:300]}, n)
            if self.cfg["g4"] and not res.success:
                self.stats["checks_run"]["g4"] = self.stats["checks_run"].get("g4", 0) + 1
                h, pat = gis_errors.hint(res.stderr)
                if h:
                    gis_notes.append("Hint: " + h)
                    self._trigger("g4", "hint", {"pattern": pat[:80]}, n)
            if gis_notes:
                obs += "\n\nGIS checks:\n- " + "\n- ".join(cap(x, 700) for x in gis_notes[:6])
        elif name == "lookup_api":
            self.stats["checks_run"]["g6_lookup"] = self.stats["checks_run"].get("g6_lookup", 0) + 1
            self._trigger("g6", "lookup", {"query": args.get("query", "")[:100]}, n)
        return obs, {"res": res}

    def _write_provenance(self):
        """G7: for every deliverable, the rounds that wrote it and the inputs read up to then."""
        path = os.path.join(self.cell_dir, "provenance.jsonl")
        if not os.path.exists(path):
            return
        rounds = [json.loads(l) for l in open(path)]
        pred = os.path.join(self.sb.work_dir, "pred_results")
        outs = sorted(f for f in os.listdir(pred) if not f.startswith(".")) if os.path.isdir(pred) else []
        lineage = {}
        for f in outs:
            writes = [r["round"] for r in rounds if any(e["mode"] == "w" and os.path.basename(e["path"]) == f for e in r["io"])]
            last = max(writes) if writes else None
            reads = sorted({e["path"] for r in rounds if last is not None and r["round"] <= last
                            for e in r["io"] if e["mode"] == "r" and e["path"].startswith("dataset")})
            lineage[f] = {"written_in_rounds": writes, "inputs_read_before": reads,
                          "code_rounds": [r["round"] for r in rounds if last is not None and r["round"] <= last and r["success"]]}
        with open(os.path.join(self.cell_dir, "provenance.json"), "w") as fh:
            json.dump({"outputs": lineage, "rounds": len(rounds)}, fh, indent=1)

    def _finish(self, n, req):
        problems, warns = [], []
        rejections = self.stats["finish_rejections"]
        pred = os.path.join(self.sb.work_dir, "pred_results")
        present = sorted(f for f in os.listdir(pred) if not f.startswith(".")) if os.path.isdir(pred) else []
        if self.cfg["generic"]:
            if not present:
                problems.append("pred_results/ is empty: no deliverable has been saved.")
            missing = [r for r in req if r not in present]
            if missing:
                problems.append(f"The task asks for these files, which are not in pred_results/: {missing}")
            if problems:
                self._trigger("h1", "finish_check", {"problems": problems}, n)
        if self.cfg["g3"] and present:
            r = self.sb.system_exec(VALIDATE_CODE, timeout=300)
            try:
                issues = json.loads((r or {}).get("stdout", "").strip().splitlines()[-1])
            except Exception:
                issues = []
            self.stats["checks_run"]["g3"] = self.stats["checks_run"].get("g3", 0) + 1
            self._log("checks.jsonl", {"round": n, "comp": "g3", "files": present, "issues": issues[:20],
                                       "parsed": r is not None})
            for it in issues:
                (problems if it["level"] == "error" else warns).append(f"{it['file']}: {it['msg']}")
            if issues:
                self._trigger("g3", "validate", {"issues": issues[:20]}, n)
        limit_reached = rejections >= 2
        if (problems or (warns and rejections == 0)) and not limit_reached:
            self.stats["finish_rejections"] += 1
            msg = "finish was not accepted yet. Please fix:\n- " + "\n- ".join(problems + warns)
            if warns and not problems:
                msg = ("Before finishing, check these points (call finish again if they are intended):\n- "
                       + "\n- ".join(warns))
            return msg + f"\nFiles in pred_results/: {present}", {}
        return f"Task marked complete. Files in pred_results/: {present}", {"finished": True}

    # ------------------------------------------------------------------ messages
    def _round_msgs(self, reply, act, obs):
        # every tool call in the reply gets a result (the API requires it); only the first was run
        calls = reply.tool_calls
        out = [{"role": "assistant", "content": reply.content or None, "_raw": _raw(reply),
                "tool_calls": [{"id": t["id"], "type": "function",
                                "function": {"name": t["name"], "arguments": t["arguments"] or "{}"}} for t in calls]}]
        for t in calls:
            out.append({"role": "tool", "tool_call_id": t["id"],
                        "content": obs if t.get("id") == act.call_id else "Not run: one tool call per reply."})
        return out
