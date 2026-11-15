"""Tier C0 (one-shot code generation baseline), the frozen prompts of the loop tiers, and
one cell run end to end through run_cell.py.
    python3 -m pytest -q harness/tests/test_oneshot.py
Scripted model replies, real container; no model is called.
"""
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402

sys.path.insert(0, paths.AGENT_DIR)
from container_sandbox import ContainerSandbox  # noqa: E402
from llm import Reply  # noqa: E402
import loop  # noqa: E402
import oneshot  # noqa: E402
import prompts  # noqa: E402
import tools  # noqa: E402

TMP_ROOT = os.environ.get("GISCLAW_TEST_TMP", tempfile.gettempdir())
TASK = {"instruction": "Test task. Save results to pred_results/out.csv.", "dataset_description": "Points.",
        "domain_knowledge": "None."}
FIXTURE = os.path.join(HERE, "fixtures", "prompt_sha.json")


def inputs(task_id):
    return os.path.join(paths.INPUTS, str(task_id), "dataset")


def have_inputs(task_ids):
    return os.path.exists(paths.TASKS_CSV) and all(os.path.isdir(inputs(t)) for t in task_ids)


class ScriptChat:
    def __init__(self, replies):
        self.replies, self.i, self.llm_time, self.stats, self.seen = list(replies), 0, 0.0, {"calls": 0}, []

    def chat(self, messages, tools=None, kind="agent"):
        self.seen.append((messages, tools, kind))
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        self.stats["calls"] += 1
        return r if isinstance(r, Reply) else Reply(r)


@pytest.fixture
def env():
    base = tempfile.mkdtemp(prefix="c0test_", dir=TMP_ROOT)
    ds = os.path.join(base, "dataset")
    os.makedirs(ds)
    with open(os.path.join(ds, "pts.csv"), "w") as f:
        f.write("x,y\n1,2\n3,4\n")
    os.chmod(ds, 0o555)
    made = []

    def run(replies, timeout=30, budget_s=1200):
        cell = tempfile.mkdtemp(prefix="cell_", dir=base)
        work = os.path.join(cell, "work")
        os.makedirs(work)
        sb = ContainerSandbox(work, timeout=timeout, dataset_dir=ds, journal_replay=True,
                              cell_id=os.path.basename(cell), exec_log=os.path.join(cell, "exec_log.jsonl"))
        made.append(sb)
        chat = ScriptChat(replies)
        st = oneshot.OneShot(chat, sb, ds, cell, tier="C0", budget_s=budget_s, verbose=False).run(TASK)
        rounds = [json.loads(l) for l in open(os.path.join(cell, "rounds.jsonl"))]
        log = os.path.join(cell, "exec_log.jsonl")
        blocks = [json.loads(l) for l in open(log)] if os.path.exists(log) else []
        pred = os.path.join(work, "pred_results")
        outs = sorted(os.listdir(pred)) if os.path.isdir(pred) else []
        run.sb = sb
        return st, rounds, blocks, outs, chat, cell
    yield run
    for sb in made:
        sb.close()
    os.chmod(ds, 0o755)


SCRIPT = ("import pandas as pd\n\n\ndef main():\n    d = pd.read_csv('dataset/pts.csv')\n"
          "    d['s'] = d.x + d.y\n    d.to_csv('pred_results/out.csv', index=False)\n    print(d.s.sum())\n\n\n"
          "if __name__ == '__main__':\n    main()\n")


def test_c0_runs_script_once_and_writes_outputs(env):
    reply = ("Plan first.\n```bash\npip install nothing\n```\nDraft:\n```python\nprint('draft')\n```\n"
             "Final:\n```python\n" + SCRIPT + "```\n")
    st, rounds, blocks, outs, chat, cell = env([reply])
    assert chat.stats["calls"] == 1 and chat.seen[0][1] is None          # one call, no tools sent
    assert [m["role"] for m in chat.seen[0][0]] == ["system", "user"]
    assert outs == ["out.csv"]
    assert len(blocks) == 1 and blocks[0]["success"]                    # exactly one exec_log block
    assert blocks[0]["code"] == oneshot.RUN_HEADER + SCRIPT              # the last python block, verbatim
    assert st["rounds"] == 1 and st["self_corrections"] == 0 and st["finished"] is True
    assert st["oneshot_no_code"] is False and st["oneshot"]["code_rule"] == "last_python"
    assert len(rounds) == 1 and "10" in rounds[0]["observation"] and rounds[0]["reply"] == reply
    p = json.load(open(os.path.join(cell, "prompt.json")))
    assert p["system"] == oneshot.system_prompt() and "Data profile" not in p["task"]
    assert p["task"] == prompts.task_message(TASK, tools.list_files(os.path.join(os.path.dirname(cell), "dataset")))


def test_c0_no_code_block_is_not_executed(env):
    st, rounds, blocks, outs, chat, cell = env(["I would read the CSV with pandas and sum x and y."])
    assert chat.stats["calls"] == 1
    assert blocks == [] and outs == []
    assert st["oneshot_no_code"] is True and st["finished"] is False and st["end_reason"] == "no_code"
    assert st["rounds"] == 1 and st["exec_blocks"] == 0
    assert rounds[0]["action"] is None


def test_c0_failing_script_gets_no_feedback_or_retry(env):
    st, rounds, blocks, outs, chat, cell = env(["```python\nraise ValueError('boom')\n```", "```python\nprint(1)\n```"])
    assert chat.stats["calls"] == 1 and len(blocks) == 1 and not blocks[0]["success"]
    assert st["finished"] is True and st["exec_failed"] == 1 and outs == []


def test_c0_empty_reply_asked_once_more(env):
    st, rounds, blocks, outs, chat, cell = env(["", "```python\nopen('pred_results/a.txt','w').write('1')\n```"])
    assert chat.stats["calls"] == 2 and st["empty_retries"] == 1 and outs == ["a.txt"]


def test_extract_code_rules():
    ex = oneshot.extract_code
    assert ex("no code") == (None, None, 0)
    assert ex("```python\na=1\n```\n```py\nb=2\n```")[0:2] == ("b=2\n", "last_python")
    assert ex("```\nshort\n```\n```text\nmuch longer block\n```")[0:2] == ("much longer block\n", "longest_fence")
    assert ex("```python\nx = 1\n# cut off by the token limit")[0] is None    # unclosed fence: not code
    assert ex("```Python\nz=3\n```")[0] == "z=3\n"


@pytest.mark.skipif(not have_inputs(json.load(open(FIXTURE))["tasks"]),
                    reason="needs benchmark/tasks.csv and the inputs of the fixture's tasks")
def test_loop_tier_prompts_are_frozen():
    """System prompt and task message of the loop tiers are byte-identical to the frozen
    versions (fixtures/prompt_sha.json: SHA-256 per task and tier)."""
    fx = json.load(open(FIXTURE))
    rows = {int(r["id"]): r for r in csv.DictReader(open(paths.TASKS_CSV, encoding="utf-8"))}
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()  # noqa: E731
    n = 0
    for key, want in fx["sha256"].items():
        t, tier = key.split("/")
        r = rows[int(t[1:])]
        task = {"instruction": r["Instruction"], "dataset_description": r["Dataset Description"],
                "domain_knowledge": r["Domain Knowledge"]}
        cfg = loop.tier_config(tier)
        if cfg.get("no_dd"):
            task["dataset_description"] = ""
        if cfg.get("no_dk"):
            task["domain_knowledge"] = ""
        files = tools.list_files(inputs(t[1:]))
        assert sha(prompts.system_prompt(cfg["tools"], cfg["generic"], cfg["g5"])) == want["system"], key
        assert sha(prompts.task_message(task, files, "DATA CARD" if cfg["g1"] else None)) == want["task"], key
        n += 1
    assert n == 24


def test_c0_script_gets_the_task_budget_not_one_step(env):
    """The one script runs with budget_s as its limit (here: step limit 3 s, budget 30 s)."""
    code = "import time\ntime.sleep(6)\nopen('pred_results/slow.txt', 'w').write('ok')\n"
    st, rounds, blocks, outs, chat, cell = env(["```python\n" + code + "```"], timeout=3, budget_s=30)
    assert env.sb.timeout == 30 and st["oneshot"]["timeout_s"] == 30
    assert blocks[0]["success"] and not blocks[0]["timed_out"] and outs == ["slow.txt"]
    st, rounds, blocks, outs, chat, cell = env(["```python\nimport time\ntime.sleep(60)\n```"], timeout=3, budget_s=4)
    assert blocks[0]["timed_out"] and not blocks[0]["success"]


# run_cell.py with its model client replaced by one scripted reply
DRIVER = r'''
import sys
sys.path.insert(0, sys.argv.pop(1))
import run_cell
from llm import Reply


class ScriptedClient:
    def __init__(self, *args, **kwargs):
        self.llm_time, self.stats = 0.0, {"calls": 0}

    def chat(self, messages, tools=None, kind="agent"):
        self.stats["calls"] += 1
        return Reply("```python\nimport os\nopen('pred_results/files.txt', 'w').write(str(len(os.listdir('dataset'))))\n```")


run_cell.ChatClient = ScriptedClient
sys.exit(run_cell.main())
'''


@pytest.mark.skipif(not have_inputs([1]), reason="needs benchmark/tasks.csv and the inputs of task 1")
def test_run_cell_records_the_c0_limit_for_the_replay():
    """run_cell records config.step_timeout_s = task_wall_s for a C0 cell (the limit its one
    script ran with) and keeps the agent tiers' limit as agent_step_timeout_s."""
    import yaml
    harness = os.path.dirname(HERE)
    cfg = yaml.safe_load(open(os.path.join(harness, "config.yaml")))
    results = tempfile.mkdtemp(prefix="c0cell_", dir=TMP_ROOT)
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", DRIVER, harness,
                        "--exp", "pytest", "--tier", "C0", "--arch", "SA", "--model", "gptoss-120b",
                        "--run", "1", "--task", "1"],
                       capture_output=True, text=True, timeout=600, env=dict(os.environ, GISCLAW_RESULTS=results))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    cell = os.path.join(results, "pytest", "C0", "SA", "gptoss-120b", "run1", "T01")
    res = json.load(open(os.path.join(cell, "cell_result.json")))
    assert res["config"]["step_timeout_s"] == cfg["task_wall_s"] == 1200
    assert res["config"]["agent_step_timeout_s"] == cfg["step_timeout_s"]
    assert res["agent_stats"]["oneshot"]["timeout_s"] == 1200 and res["oneshot_no_code"] is False
    assert res["status"] == "done" and res["outputs"] == ["files.txt"] and res["deliverable_produced"] is True
    assert res["inputs_intact"] is True and res["protocol"] == "native"
    assert sorted(os.listdir(os.path.join(cell, "gptoss-120b"))) == ["code.py", "files.txt"]
