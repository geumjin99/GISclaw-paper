"""Claude (Anthropic Messages API) client with the same interface as llm.ChatClient.

The agent loop keeps OpenAI-style messages; this client converts them per request:
  - the system message becomes the top-level `system`;
  - an assistant message carries the provider's own content blocks in `_raw` (thinking,
    text, tool_use) and is sent back unchanged - current Claude models require an
    append-only history with their blocks passed back as returned;
  - consecutive `tool` messages become one user message of `tool_result` blocks.
Request settings come from the experiment config (`set` / `drop`), e.g. effort via
output_config. No server-side model fallback: a refusal is recorded, never answered by a
different model, so every result belongs to the model under test.
"""
import hashlib
import json
import time

from llm import CONTEXT_MARKERS, ContextOverflow, LLMFailure, Reply


def _blocks_to_dicts(content):
    out = []
    for b in content:
        try:
            d = b.model_dump(exclude_none=True)
        except Exception:
            d = dict(b)
        out.append(d)
    return out


class ClaudeChat:
    def __init__(self, model, api_key=None, request=None, timeout=900, record_path=None):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=3)
        self.model = model
        req = request or {}
        self.set = dict(req.get("set") or {})
        self.drop = set(req.get("drop") or [])
        self.record_path = record_path
        self.n = 0
        self.llm_time = 0.0
        self.stats = {"calls": 0, "retries": 0, "errors": 0, "context_overflow": 0, "empty": 0, "refusals": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cache_write_tokens": 0}

    # ------------------------------------------------------------------ conversion
    @staticmethod
    def convert(messages):
        system, out = None, []
        for m in messages:
            role = m["role"]
            if role == "system":
                system = m["content"]
            elif role == "assistant":
                if m.get("_raw") is not None:
                    content = m["_raw"]
                else:
                    content = [{"type": "text", "text": m.get("content") or "(empty)"}]
                    for tc in m.get("tool_calls") or []:
                        fn = tc.get("function", tc)
                        args = fn.get("arguments") or "{}"
                        content.append({"type": "tool_use", "id": tc["id"], "name": fn["name"],
                                        "input": json.loads(args) if isinstance(args, str) else args})
                out.append({"role": "assistant", "content": content})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m.get("content") or ""}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][-1].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            else:
                out.append({"role": "user", "content": m.get("content") or ""})
        return system, out

    @staticmethod
    def tool_defs(openai_tools):
        return [{"name": t["function"]["name"], "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]} for t in openai_tools or []]

    # ------------------------------------------------------------------ call
    def chat(self, messages, tools=None, kind="agent"):
        system, msgs = self.convert(messages)
        params = {"model": self.model, "max_tokens": 32000, "messages": msgs,
                  "cache_control": {"type": "ephemeral"}}
        if system:
            params["system"] = system
        if tools:
            params["tools"] = self.tool_defs(tools)
            params["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}  # one action per reply
        params.update(self.set)
        for k in self.drop:
            params.pop(k, None)
        prompt_chars = len(json.dumps(msgs, ensure_ascii=False, default=str)) + len(system or "")
        A = self.anthropic
        last = None
        for attempt, wait in enumerate((0, 10, 30, 90)):
            if wait:
                self.stats["retries"] += 1
                time.sleep(wait)
            t0 = time.time()
            try:
                with self.client.messages.stream(**params) as s:
                    msg = s.get_final_message()
                dt = time.time() - t0
                self.llm_time += dt
                return self._finish(msg, kind, params, prompt_chars, dt, attempt)
            except A.BadRequestError as e:
                dt = time.time() - t0
                self.llm_time += dt
                text = str(e)
                if any(m in text.lower() for m in CONTEXT_MARKERS):
                    self.stats["context_overflow"] += 1
                    self._record({"kind": kind, "error": "context_overflow", "detail": text[:500], "prompt_chars": prompt_chars})
                    raise ContextOverflow(text[:300])
                self._record({"kind": kind, "error": "bad_request", "detail": text[:800], "prompt_chars": prompt_chars})
                last = f"BadRequest: {text[:300]}"
                break  # will not succeed on retry
            except (A.RateLimitError, A.InternalServerError, A.APIConnectionError, A.APITimeoutError) as e:
                dt = time.time() - t0
                self.llm_time += dt
                last = f"{type(e).__name__}: {str(e)[:300]}"
                self._record({"kind": kind, "error": "transient", "detail": last, "attempt": attempt})
            except A.APIStatusError as e:
                dt = time.time() - t0
                self.llm_time += dt
                last = f"{type(e).__name__} {e.status_code}: {str(e)[:300]}"
                self._record({"kind": kind, "error": "status", "detail": last, "attempt": attempt})
                if e.status_code < 500:
                    break
        self.stats["errors"] += 1
        raise LLMFailure(last)

    def _finish(self, msg, kind, params, prompt_chars, dt, attempt):
        raw = _blocks_to_dicts(msg.content)
        text = "\n".join(b.get("text", "") for b in raw if b.get("type") == "text").strip()
        thinking = "\n".join(b.get("thinking", "") for b in raw if b.get("type") == "thinking").strip()
        tcs = [{"id": b["id"], "name": b["name"], "arguments": json.dumps(b.get("input") or {})}
               for b in raw if b.get("type") == "tool_use"]
        u = msg.usage
        usage = {"prompt_tokens": getattr(u, "input_tokens", 0), "completion_tokens": getattr(u, "output_tokens", 0),
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += usage["prompt_tokens"] or 0
        self.stats["completion_tokens"] += usage["completion_tokens"] or 0
        self.stats["cached_tokens"] += usage["cache_read_input_tokens"]
        self.stats["cache_write_tokens"] += usage["cache_creation_input_tokens"]
        if msg.stop_reason == "refusal":
            self.stats["refusals"] += 1
        if not text and not tcs:
            self.stats["empty"] += 1
        self._record({
            "kind": kind, "attempt": attempt, "latency_s": round(dt, 2),
            "request": {k: v for k, v in params.items() if k not in ("messages", "tools", "system")},
            "n_messages": len(params["messages"]), "prompt_chars": prompt_chars,
            "messages_sha": hashlib.sha256(json.dumps(params["messages"], ensure_ascii=False, default=str).encode()).hexdigest()[:16],
            "finish_reason": msg.stop_reason,
            "stop_details": (msg.stop_details.model_dump() if getattr(msg, "stop_details", None) else None),
            "usage": usage, "content": text, "reasoning": thinking, "tool_calls": tcs,
            "served_model": msg.model,
        })
        r = Reply(text, tcs, thinking, msg.stop_reason, usage)
        r.raw = raw
        return r

    def _record(self, rec):
        self.n += 1
        rec = {"i": self.n, "t": round(time.time(), 3), **rec}
        if self.record_path:
            with open(self.record_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
