"""Planner-worker (dual) agent on the same core as the single agent.

Same model, tools, guards, validation, prompts, round pool (worker rounds only) and
agent-time budget as the single agent. The only difference is the structure:
  plan (after the data card) -> per step: worker ReAct -> verifier -> one retry with the
  verifier's hint -> replan (at most 2) -> final deliverable check.
Every worker step sees the task, the data card, the full plan, the verdicts so far, the
current variables and the files in pred_results/.
"""
import math

from history import History
from llm import ContextOverflow, LLMFailure
from loop import Agent
from observe import cap
from protocol import _decode_object

PLANNER_SYS = """You plan geospatial data analysis tasks for an assistant that writes and runs Python code.
Split the task into 3-7 steps. Each step must produce something checkable (a variable, a printed number, or a file).
The last step must save every deliverable the task asks for in pred_results/.
Reply with one JSON object only:
{"steps": [{"id": 1, "goal": "...", "produces": "variables / files / printed values this step must leave behind"}]}"""

REPLAN_NOTE = """The plan below did not work. Done so far:
{done}
Variables now in the environment: {vars}
Files in pred_results/: {files}
Problem: {problem}
Write a new plan for the REMAINING work only (same JSON format)."""

VERIFIER_SYS = """You check whether one step of a data analysis was completed. Judge only from the evidence given
(the step's actions and outputs, the variables and files that exist). Reply with one JSON object only:
{"pass": true or false, "reason": "one sentence", "fix_hint": "what to do next if not passed, else empty"}"""

STEP_NOTE = """You are carrying out one step of this plan:
{plan}

Current step: {cur}
Verdicts on earlier steps:
{verdicts}
Variables now in the environment: {vars}
Files in pred_results/: {files}

Do only the current step, then call finish with a one-line summary of what it produced.{final}"""

VARS_CODE = r'''
def _gg_vars():
    import numpy as _np, pandas as _pd
    out = []
    for k, v in list(globals().items()):
        if k.startswith('_') or callable(v) or type(v).__name__ == 'module':
            continue
        try:
            import geopandas as _gpd
            if isinstance(v, _gpd.GeoDataFrame):
                out.append(f"{k}: GeoDataFrame {len(v)} rows, CRS={v.crs.to_string() if v.crs is not None else None}"); continue
        except Exception:
            pass
        if isinstance(v, _pd.DataFrame):
            out.append(f"{k}: DataFrame {v.shape}")
        elif isinstance(v, _np.ndarray):
            out.append(f"{k}: ndarray {v.shape} {v.dtype}")
        elif isinstance(v, (int, float, str, bool)):
            out.append(f"{k}: {type(v).__name__} = {str(v)[:40]}")
        else:
            out.append(f"{k}: {type(v).__name__}")
    return out
print("; ".join(_gg_vars()[:60]) or "(none)")
del _gg_vars
'''


def _json(text):
    try:
        obj, _ = _decode_object(text or "")
        return obj if isinstance(obj, dict) else None
    except ValueError:
        return None


class DualAgent(Agent):
    def __init__(self, *a, max_replans=2, **kw):
        super().__init__(*a, **kw)
        self.max_replans = max_replans
        self.stats.update(planner_calls=0, verifier_calls=0, replans=0, verify_unknown=0, steps=[],
                          plan_fallback=False)

    # ------------------------------------------------------------------ helpers
    def _vars(self):
        r = self.sb.system_exec(VARS_CODE, timeout=60)
        return cap(((r or {}).get("stdout") or "").strip() or "(unavailable)", 2500)

    def _files(self):
        import os
        pred = os.path.join(self.sb.work_dir, "pred_results")
        return sorted(f for f in os.listdir(pred) if not f.startswith(".")) if os.path.isdir(pred) else []

    def _plan(self, extra=None):
        msgs = [{"role": "system", "content": PLANNER_SYS},
                {"role": "user", "content": self.user + (("\n\n" + extra) if extra else "")}]
        for attempt in range(3):
            self.stats["planner_calls"] += 1
            r = self.llm.chat(msgs, kind="planner")
            obj = _json(r.content)
            steps = (obj or {}).get("steps")
            if isinstance(steps, list) and steps and all(isinstance(s, dict) and s.get("goal") for s in steps):
                return [{"id": i + 1, "goal": str(s["goal"]), "produces": str(s.get("produces", ""))}
                        for i, s in enumerate(steps[:7])]
            msgs += [{"role": "assistant", "content": r.content or "(empty)"},
                     {"role": "user", "content": 'That was not a valid plan. Reply with one JSON object: '
                                                 '{"steps": [{"id": 1, "goal": "...", "produces": "..."}]}'}]
        self.stats["plan_fallback"] = True
        return [{"id": 1, "goal": "Complete the whole task as described.", "produces": "all deliverables"}]

    def _verify(self, step, trace, vars_before, vars_after, files_before, files_after):
        ev = (f"Step: {step['goal']}\nExpected: {step['produces']}\n\nActions and outputs:\n{trace}\n\n"
              f"Variables before: {vars_before}\nVariables after: {vars_after}\n"
              f"Files before: {files_before}\nFiles after: {files_after}")
        msgs = [{"role": "system", "content": VERIFIER_SYS}, {"role": "user", "content": cap(ev, 12000, keep="tail")}]
        self.stats["verifier_calls"] += 1
        r = self.llm.chat(msgs, kind="verifier")
        obj = _json(r.content)
        if not obj or "pass" not in obj:
            self.stats["verify_unknown"] += 1
            return {"pass": None, "reason": "verifier reply could not be read", "fix_hint": ""}
        return {"pass": bool(obj.get("pass")), "reason": str(obj.get("reason", ""))[:300],
                "fix_hint": str(obj.get("fix_hint", ""))[:500]}

    def _trace(self, hist, since):
        lines = []
        for r in hist.rounds:
            if (r.get("n") or 0) <= since:
                continue
            for m in r["msgs"]:
                c = m.get("content") or ""
                if m["role"] in ("user", "tool"):
                    lines.append("Result: " + cap(c, 900, keep="tail"))
                elif m["role"] == "assistant":
                    lines.append("Action: " + cap(c, 700))
                    for tc in m.get("tool_calls") or []:
                        lines.append("Call: " + cap(str(tc.get("function")), 700))
        return "\n".join(lines[-16:]) or "(no actions)"

    # ------------------------------------------------------------------ run
    def run(self, task):
        self.setup(task)
        try:
            plan = self._plan()
        except (LLMFailure, ContextOverflow) as e:
            self.stats["end_reason"] = f"llm_error: planner {str(e)[:150]}"
            return self.wrapup(False, [])
        hists, verdicts, done = [], [], []
        i = 0
        finished = False
        while i < len(plan):
            if self.n >= self.max_rounds or self.agent_time() > self.budget_s:
                break
            step = plan[i]
            final = i == len(plan) - 1
            remaining = self.max_rounds - self.n
            reserve = 0 if final else 5
            soft = max(5, math.ceil((remaining - reserve) / (len(plan) - i)) + 3)
            plan_txt = "\n".join(f"{'->' if s is step else '  '} {s['id']}. {s['goal']} (produces: {s['produces']})"
                                 for s in plan)
            vb, fb = self._vars(), self._files()
            note = STEP_NOTE.format(plan=plan_txt, cur=f"{step['id']}. {step['goal']}",
                                    verdicts="\n".join(verdicts) or "(none yet)", vars=vb, files=fb,
                                    final=("\nThis is the last step: every deliverable must be saved in pred_results/ "
                                           "before you call finish." if final else ""))
            hist = History(self.sysmsg, self.user + "\n\n" + note, self.history_tokens)
            hists.append(hist)
            start = self.n
            ok, summary = self.react(hist, soft, final=final, kind="worker")
            if self.stats["end_reason"] and str(self.stats["end_reason"]).startswith(("llm_error", "context", "format", "budget")):
                break
            if final and ok:
                finished = True
                done.append(f"{step['id']}. {step['goal']}: {summary}")
                break
            try:
                v = self._verify(step, self._trace(hist, start), vb, self._vars(), fb, self._files())
                if v["pass"] is not True and self.n < self.max_rounds:
                    hint = v["fix_hint"] or v["reason"]
                    # keep user/assistant alternation: the last message answered the worker's finish
                    hist.add([{"role": "assistant", "content": "(step reported complete)"},
                              {"role": "user", "content": f"A reviewer checked this step: not done yet. {v['reason']} "
                                                          f"What to do: {hint}"}], n=self.n)
                    ok2, summary = self.react(hist, max(3, soft // 2), final=final, kind="worker_retry")
                    if final and ok2:
                        finished = True
                        break
                    v = self._verify(step, self._trace(hist, start), vb, self._vars(), fb, self._files())
            except (LLMFailure, ContextOverflow) as e:
                self.stats["end_reason"] = f"llm_error: verifier {str(e)[:150]}"
                break
            self.stats["steps"].append({"id": step["id"], "goal": step["goal"][:200], "rounds": self.n - start,
                                        "pass": v["pass"], "reason": v["reason"][:200]})
            verdicts.append(f"{step['id']}. {'passed' if v['pass'] else 'NOT passed'}: {v['reason']}")
            done.append(f"{step['id']}. {step['goal']} -> {'ok' if v['pass'] else 'not ok'}: {summary or v['reason']}")
            if v["pass"] is not True and self.stats["replans"] < self.max_replans and self.n < self.max_rounds - 5:
                self.stats["replans"] += 1
                try:
                    new = self._plan(REPLAN_NOTE.format(done="\n".join(done), vars=self._vars(), files=self._files(),
                                                        problem=v["reason"]))
                except (LLMFailure, ContextOverflow) as e:
                    self.stats["end_reason"] = f"llm_error: replanner {str(e)[:150]}"
                    break
                plan = plan[:i + 1] + [dict(s, id=i + 2 + k) for k, s in enumerate(new)]
            i += 1
        # deliverables still missing after the plan: one last saving step from the reserved rounds
        if not finished and self.n < self.max_rounds and self.agent_time() < self.budget_s \
                and not str(self.stats.get("end_reason") or "").startswith(("llm_error", "context", "format")):
            missing = [r for r in self.req if r not in self._files()] if self.req else ([] if self._files() else ["(any)"])
            note = (f"The plan is finished. Save the deliverables now (missing: {missing}), check them, then call finish."
                    if missing else "The plan is finished. Check the deliverables in pred_results/, then call finish.")
            hist = History(self.sysmsg, self.user + "\n\n" + note + f"\nVariables now: {self._vars()}", self.history_tokens)
            hists.append(hist)
            ok, _ = self.react(hist, self.max_rounds - self.n, final=True, kind="worker_final")
            finished = ok
        return self.wrapup(finished, hists)
