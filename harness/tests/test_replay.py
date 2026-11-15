"""checked_code_text: a failed block whose observation lost the head of the traceback
(tail-truncated stderr) keeps its completed prefix when the full traceback of the scoring
replay is given. No container needed.

    python3 -m pytest -q harness/tests/test_replay.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from score import checked_code_text  # noqa: E402

CODE = "from sklearn.ensemble import RandomForestClassifier\nrf = RandomForestClassifier(5)\nrf.fit(X, y)\nbad_call()\n"
FULL_TB = ('Traceback (most recent call last):\n  File "<string>", line 4, in <module>\n'
           '  File "/lib/x.py", line 9, in f\nValueError: boom')
TRUNC_OBS = "Execution failed\nstderr:\n... (855 characters omitted)\n  File \"/lib/x.py\", line 9, in f\nValueError: boom"


def _cell(tmp_path, obs):
    with open(tmp_path / "rounds.jsonl", "w") as f:
        f.write(json.dumps({"action": "execute", "args": {"code": "a = 1"}, "observation": "ok"}) + "\n")
        f.write(json.dumps({"action": "execute", "args": {"code": CODE}, "observation": obs}) + "\n")
    return [{"code": "a = 1", "success": True}, {"code": CODE, "success": False}]


def test_truncated_observation_uses_replay_traceback(tmp_path):
    entries = _cell(tmp_path, TRUNC_OBS)
    txt, n = checked_code_text(str(tmp_path), entries)
    assert n == 0 and "RandomForestClassifier(5)" not in txt          # the failing line cannot be located
    txt, n = checked_code_text(str(tmp_path), entries, replay_tb=[None, FULL_TB])
    assert n == 1 and "RandomForestClassifier(5)" in txt and "bad_call" not in txt


def test_different_error_in_replay_not_used(tmp_path):
    # the run died earlier (MemoryError, truncated); the replay failed further down: no prefix credited
    entries = _cell(tmp_path, "Execution failed\nstderr:\n... (900 characters omitted)\nMemoryError")
    txt, n = checked_code_text(str(tmp_path), entries, replay_tb=[None, FULL_TB])
    assert n == 0 and "RandomForestClassifier(5)" not in txt


def test_full_observation_unchanged(tmp_path):
    entries = _cell(tmp_path, "Execution failed\nstderr:\n" + FULL_TB)
    a = checked_code_text(str(tmp_path), entries)
    b = checked_code_text(str(tmp_path), entries, replay_tb=[None, FULL_TB])
    assert a == b and a[1] == 1
