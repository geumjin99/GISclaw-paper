"""Chat client for OpenAI-compatible endpoints (e.g. a local vLLM server).

- Real multi-turn messages (system / user / assistant / tool), not one flattened prompt.
- Request parameters come from the experiment config: `set` adds/overrides, `drop` removes.
  Nothing is inferred from the model name.
- Every call is recorded with its full reply text, reasoning text, tool calls, finish
  reason, usage and the call kind (agent / planner / verifier / ...).
- Transport failures are retried invisibly; a context overflow is reported as such.
"""
import hashlib
import json
import time

TRANSIENT_WAITS = (5, 20, 60)
CONTEXT_MARKERS = ("maximum context length", "context length", "too many tokens", "prompt is too long",
                   "context_length_exceeded", "input is too long", "reduce the length")


class ContextOverflow(RuntimeError):
    pass


class LLMFailure(RuntimeError):
    pass


class Reply:
    def __init__(self, content="", tool_calls=None, reasoning="", finish_reason=None, usage=None):
        self.content = content or ""
        self.tool_calls = tool_calls or []        # [{"id", "name", "arguments"}]
        self.reasoning = reasoning or ""
        self.finish_reason = finish_reason
        self.usage = usage or {}


def _usage(u):
    if u is None:
        return {}
    try:
        d = u.model_dump()
    except Exception:
        d = dict(getattr(u, "__dict__", {}))
    return json.loads(json.dumps(d, default=str))


class ChatClient:
    def __init__(self, base_url, model, api_key="EMPTY", request=None, timeout=600, record_path=None):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
        self.model = model
        req = request or {}
        self.set = dict(req.get("set") or {})
        self.drop = set(req.get("drop") or [])
        self.record_path = record_path
        self.n = 0
        self.llm_time = 0.0
        self.stats = {"calls": 0, "retries": 0, "errors": 0, "context_overflow": 0, "empty": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cached_tokens": 0}

    def _params(self, messages, tools):
        # keys starting with "_" are harness bookkeeping (e.g. another provider's raw blocks)
        p = {"model": self.model, "messages": [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]}
        p.update(self.set)
        for k in self.drop:
            p.pop(k, None)
        if tools:
            p["tools"] = tools
            p["tool_choice"] = "auto"
        return p

    def chat(self, messages, tools=None, kind="agent"):
        params = self._params(messages, tools)
        prompt_chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
        last_err = None
        for attempt, wait in enumerate((0,) + TRANSIENT_WAITS):
            if wait:
                self.stats["retries"] += 1
                time.sleep(wait)
            t0 = time.time()
            try:
                r = self.client.chat.completions.create(**params)
                dt = time.time() - t0
                self.llm_time += dt
                return self._finish(r, kind, params, prompt_chars, dt, attempt)
            except Exception as e:  # noqa: BLE001 - classify below
                dt = time.time() - t0
                self.llm_time += dt
                msg = str(e)
                status = getattr(e, "status_code", None)
                if any(m in msg.lower() for m in CONTEXT_MARKERS):
                    self.stats["context_overflow"] += 1
                    self._record({"kind": kind, "error": "context_overflow", "detail": msg[:500],
                                  "prompt_chars": prompt_chars, "latency_s": round(dt, 2)})
                    raise ContextOverflow(msg[:300])
                last_err = f"{type(e).__name__} {status}: {msg[:300]}"
                self._record({"kind": kind, "error": "transient", "detail": last_err,
                              "prompt_chars": prompt_chars, "latency_s": round(dt, 2), "attempt": attempt})
                if status is not None and 400 <= status < 500 and status not in (408, 409, 429):
                    break  # a request the server will never accept: do not retry
        self.stats["errors"] += 1
        raise LLMFailure(last_err)

    def _finish(self, r, kind, params, prompt_chars, dt, attempt):
        ch = r.choices[0]
        msg = ch.message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        tcs = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = getattr(tc, "function", None)
            tcs.append({"id": getattr(tc, "id", None), "name": getattr(fn, "name", None),
                        "arguments": getattr(fn, "arguments", None)})
        usage = _usage(getattr(r, "usage", None))
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += usage.get("prompt_tokens") or 0
        self.stats["completion_tokens"] += usage.get("completion_tokens") or 0
        self.stats["reasoning_tokens"] += ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        self.stats["cached_tokens"] += ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        if not content.strip() and not tcs:
            self.stats["empty"] += 1
        self._record({
            "kind": kind, "attempt": attempt, "latency_s": round(dt, 2),
            "request": {k: v for k, v in params.items() if k not in ("messages", "tools")},
            "n_messages": len(params["messages"]), "prompt_chars": prompt_chars,
            "messages_sha": hashlib.sha256(json.dumps(params["messages"], ensure_ascii=False).encode()).hexdigest()[:16],
            "finish_reason": ch.finish_reason, "usage": usage,
            "content": content, "reasoning": reasoning, "tool_calls": tcs,
            "refusal": getattr(msg, "refusal", None), "served_model": getattr(r, "model", None),
        })
        return Reply(content, tcs, reasoning, ch.finish_reason, usage)

    def _record(self, rec):
        self.n += 1
        rec = {"i": self.n, "t": round(time.time(), 3), **rec}
        if self.record_path:
            with open(self.record_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
