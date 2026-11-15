"""Conversation history with a token budget (all tiers).

The history is a list of rounds; each round is the assistant message plus the message(s)
that answer it (the tool results). When the estimated prompt exceeds the budget, the oldest
rounds that are not pinned are dropped as whole rounds (so tool calls never lose their
results) and replaced by one short note.
Pinned rounds (loading / inspecting data) are kept, so column names and CRS stay visible.
"""
CHARS_PER_TOKEN = 3.2


def est_tokens(messages):
    n = 0
    for m in messages:
        c = m.get("content") or ""
        n += len(c) if isinstance(c, str) else len(str(c))
        for tc in m.get("tool_calls") or []:
            n += len(str(tc))
    return int(n / CHARS_PER_TOKEN) + 4 * len(messages)


class History:
    def __init__(self, system, task_message, budget_tokens):
        self.system = {"role": "system", "content": system}
        self.task = {"role": "user", "content": task_message}
        self.budget = budget_tokens
        self.rounds = []          # dicts: {"msgs": [...], "pinned": bool, "n": round number}
        self.dropped = 0
        self.truncations = 0

    def add(self, msgs, pinned=False, n=None):
        self.rounds.append({"msgs": list(msgs), "pinned": pinned, "n": n})

    def messages(self):
        kept = list(self.rounds)
        dropped = 0
        while True:
            msgs = self._assemble(kept, dropped)
            if est_tokens(msgs) <= self.budget:
                break
            # drop the oldest unpinned round, but always keep the last 4 rounds
            idx = next((i for i, r in enumerate(kept[:-4]) if not r["pinned"]), None)
            if idx is None:
                break
            kept.pop(idx)
            dropped += 1
        if dropped > self.dropped:
            self.truncations += 1
        self.dropped = max(self.dropped, dropped)
        return msgs

    def _assemble(self, kept, dropped):
        task = self.task
        if dropped:  # appended to the task message, so user/assistant turns still alternate
            task = {"role": "user", "content": self.task["content"] +
                    f"\n\n[Note: {dropped} earlier round(s) were removed from this conversation to save space. "
                    f"Everything they created still exists in the Python environment.]"}
        out = [self.system, task]
        for r in kept:
            out.extend(r["msgs"])
        return out
