"""Turn an execution result into the observation text the model sees (all tiers).

Limits: stdout keeps its head and tail, stderr keeps its tail (the real error is at the
end of a traceback), Python warnings are de-duplicated and shown separately, and the whole
observation has a hard cap so no single step can blow up the context.
"""
import re

STDOUT_HEAD, STDOUT_TAIL = 2400, 800
STDERR_TAIL = 2000
MAX_WARNINGS, WARNING_CHARS = 5, 300
MAX_VARS, VAR_CHARS = 15, 200
HARD_CAP = 6000

_WARN = re.compile(r"^(?:.*?:\d+: )?(\w*Warning): (.*)$")


def cap(text, limit, keep="head"):
    text = text or ""
    if len(text) <= limit:
        return text
    if keep == "tail":
        return f"... ({len(text) - limit} characters omitted)\n" + text[-limit:]
    return text[:limit] + f"\n... ({len(text) - limit} characters omitted)"


def head_tail(text, head, tail):
    text = text or ""
    if len(text) <= head + tail + 80:
        return text
    return (text[:head] + f"\n... ({len(text) - head - tail} characters omitted) ...\n" + text[-tail:])


def split_warnings(stderr):
    """Separate Python warning lines (and their echoed source line) from real errors."""
    warns, rest, skip_next = [], [], False
    lines = (stderr or "").splitlines()
    for i, ln in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        m = _WARN.match(ln)
        if m and "Traceback" not in ln:
            warns.append(f"{m.group(1)}: {m.group(2).strip()}")
            # warnings.showwarning echoes the source line, indented, on the next line
            if i + 1 < len(lines) and lines[i + 1].startswith("  "):
                skip_next = True
            continue
        rest.append(ln)
    uniq = []
    for w in warns:
        if w not in uniq:
            uniq.append(w)
    return uniq, "\n".join(rest).strip()


def format_exec(res):
    """res: object with stdout, stderr, success, new_vars."""
    parts = ["\u2705 Execution successful" if res.success else "\u274c Execution failed"]
    out = (res.stdout or "").strip()
    if out:
        parts.append("stdout:\n" + head_tail(out, STDOUT_HEAD, STDOUT_TAIL))
    warns, err = split_warnings(res.stderr)
    if err:
        parts.append("stderr:\n" + cap(err, STDERR_TAIL, keep="tail"))
    if warns:
        shown = [cap(w, WARNING_CHARS) for w in warns[:MAX_WARNINGS]]
        more = f"\n  (+{len(warns) - MAX_WARNINGS} more)" if len(warns) > MAX_WARNINGS else ""
        parts.append("Warnings:\n  " + "\n  ".join(shown) + more)
    nv = getattr(res, "new_vars", None) or {}
    if nv:
        items = list(nv.items())
        lines = [f"  {k}: {cap(str(v).splitlines()[0] if str(v) else '', VAR_CHARS)}" for k, v in items[:MAX_VARS]]
        if len(items) > MAX_VARS:
            lines.append(f"  (+{len(items) - MAX_VARS} more)")
        parts.append("New variables:\n" + "\n".join(lines))
    return cap("\n".join(parts), HARD_CAP)


def last_error_line(stderr):
    _, err = split_warnings(stderr)
    lines = [l for l in err.splitlines() if l.strip()]
    return lines[-1].strip()[:300] if lines else ""
