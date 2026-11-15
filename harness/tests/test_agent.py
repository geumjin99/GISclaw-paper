"""Tests of the agent loop and the dual agent against a real container.
    python3 -m pytest -q harness/tests/test_agent.py
Each test scripts the model's replies; no model is called.
"""
import itertools
import json
import os
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

TMP_ROOT = os.environ.get("GISCLAW_TEST_TMP", tempfile.gettempdir())
TASK = {"instruction": "Test task. Save results to pred_results/out.geojson.", "dataset_description": "",
        "domain_knowledge": ""}


def pts_fc(n=30, x0=568000, y0=300000, step=500):
    return {"type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::32616"}},
            "features": [{"type": "Feature", "properties": {"v": float(i)},
                          "geometry": {"type": "Point", "coordinates": [x0 + i * step, y0 + i * step]}}
                         for i in range(n)]}


class ScriptChat:
    def __init__(self, replies):
        self.replies, self.i, self.llm_time, self.stats, self.seen = list(replies), 0, 0.0, {"calls": 0}, []

    def chat(self, messages, tools=None, kind="agent"):
        self.seen.append(messages)
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        self.stats["calls"] += 1
        return r if isinstance(r, Reply) else Reply(r)


_ids = itertools.count(1)


def call(name, args, thought="t"):
    """A reply that calls one tool."""
    return Reply(thought, [{"id": f"call_{next(_ids)}", "name": name, "arguments": json.dumps(args)}])


def ex(code, thought="t"):
    return call("execute", {"code": code}, thought)


def fin(summary="s"):
    return call("finish", {"summary": summary}, "done")


FIN = fin()


@pytest.fixture
def env():
    base = tempfile.mkdtemp(prefix="agenttest_", dir=TMP_ROOT)
    ds = os.path.join(base, "dataset")
    os.makedirs(ds)
    with open(os.path.join(ds, "pts.geojson"), "w") as f:
        json.dump(pts_fc(), f)
    os.chmod(ds, 0o555)
    made = []

    def run(replies, tier="H2", timeout=30, **kw):
        cell = tempfile.mkdtemp(prefix="cell_", dir=base)
        work = os.path.join(cell, "work")
        os.makedirs(work)
        sb = ContainerSandbox(work, timeout=timeout, dataset_dir=ds, journal_replay=True,
                              cell_id=os.path.basename(cell))
        made.append(sb)
        chat = ScriptChat(replies)
        ag = loop.Agent(chat, sb, ds, cell, tier=tier, verbose=False, **kw)
        st = ag.run(TASK)
        rounds = [json.loads(l) for l in open(os.path.join(cell, "rounds.jsonl"))]
        trig = ([json.loads(l) for l in open(os.path.join(cell, "triggers.jsonl"))]
                if os.path.exists(os.path.join(cell, "triggers.jsonl")) else [])
        return st, rounds, trig, chat, sb
    yield run
    for sb in made:
        sb.close()
    os.chmod(ds, 0o755)


def obs(rounds, n):
    return next(r for r in rounds if r.get("n") == n and "observation" in r)["observation"]


def test_format_errors_are_reported_and_never_executed(env):
    replies = [Reply("print(3)"),                                                           # text only: no tool call
               Reply("t", [{"id": "b1", "name": "execute", "arguments": '{"code": "x = 1"'}]),  # broken JSON arguments
               Reply("t", [{"id": "b2", "name": "run_shell", "arguments": '{"cmd": "ls"}'}]),   # unknown tool
               Reply("t", [{"id": "b3", "name": "execute", "arguments": "y = 2"}]),            # bare value: taken as code
               ex("import os\nos.makedirs('pred_results', exist_ok=True)\nprint(y + 1)"), FIN, FIN, FIN]
    st, rounds, trig, chat, sb = env(replies, tier="H0")
    assert st["format_errors"] == 3
    assert st["format_error_kinds"] == {"no_action": 1, "bad_args": 1, "unknown_tool": 1}
    assert "3" in obs(rounds, 2)
    codes = [c for c, _, _ in sb.journal]
    assert len(codes) == 2 and codes[0] == "y = 2"


def test_g2_empty_join_and_warnings(env):
    code = ("a = gpd.read_file('dataset/pts.geojson')\n"
            "b = a.to_crs(4326).buffer(0.0001).to_frame('geometry')\n"
            "j = gpd.sjoin(a, gpd.GeoDataFrame(b, geometry='geometry', crs=4326))\nprint(len(j))")
    st, rounds, trig, chat, sb = env([ex(code), FIN, FIN, FIN], tier="H2")
    o = obs(rounds, 1)
    assert "GIS checks" in o and "returned 0 rows" in o
    assert any(t["kind"] == "empty_result" for t in trig)
    st0, rounds0, *_ = env([ex(code), FIN, FIN, FIN], tier="H1")
    assert "GIS checks" not in obs(rounds0, 1)


def test_g2_g3_no_crs_written_and_finish_rejected(env):
    code = ("g = gpd.GeoDataFrame({'v': [1, 2]}, geometry=[Point(568000, 300000), Point(569000, 301000)])\n"
            "g.to_file('pred_results/out.geojson')")
    st, rounds, trig, chat, sb = env([ex(code), FIN, ex("g = g.set_crs(32616)\ng.to_file('pred_results/out.geojson')"),
                                      FIN, FIN], tier="H2")
    assert "without a CRS" in obs(rounds, 1)
    assert "not accepted" in obs(rounds, 2) and "saved without their CRS" in obs(rounds, 2)
    assert st["finished"] and st["finish_rejections"] == 1


def test_g2_raster_nodata_and_grid(env):
    code = ("import numpy as np, rasterio\nfrom rasterio.transform import from_origin\n"
            "a = np.full((20, 20), -9999, 'float32'); a[5:15, 5:15] = 3\n"
            "for name, res in (('a.tif', 10), ('b.tif', 20)):\n"
            "    with rasterio.open(name, 'w', driver='GTiff', height=20, width=20, count=1, dtype='float32',"
            " crs='EPSG:32616', transform=from_origin(568000, 301000, res, res), nodata=-9999) as d:\n"
            "        d.write(a, 1)\n"
            "x = rasterio.open('a.tif').read(1); y = rasterio.open('b.tif').read(1)\nprint(x.mean())")
    st, rounds, trig, chat, sb = env([ex(code), FIN, FIN, FIN], tier="H2")
    kinds = [t["kind"] for t in trig]
    assert "nodata_unmasked" in kinds and "grid_mismatch" in kinds
    # one note per file, however often or however much of it is read
    st2, rounds2, trig2, *_ = env([ex(code), ex("for _ in range(3):\n    x = rasterio.open('a.tif').read(1, window=((0, 10 + _), (0, 10)))"),
                                   FIN, FIN, FIN], tier="H2")
    assert [t["kind"] for t in trig2].count("nodata_unmasked") == 2   # a.tif and b.tif, once each


def test_g2_shapely_degrees_and_g4_hint(env):
    st, rounds, trig, chat, sb = env([
        ex("from shapely.geometry import Point, box\nsq = box(0, 0, 1, 1)\nprint(sq.area)"),   # abstract coords: no warning
        ex("ll = gpd.read_file('dataset/pts.geojson').to_crs(4326)\np = ll.geometry.iloc[0]\nprint(p.buffer(0.01).area)"),
        ex("gpd.GeoDataFrame(geometry=[Point(0, 0)]).to_crs(3857)"), FIN, FIN, FIN], tier="H2")
    assert "degrees" not in obs(rounds, 1)
    assert "degrees" in obs(rounds, 2)
    assert "Hint:" in obs(rounds, 3) and "set_crs" in obs(rounds, 3)


def test_dedup_only_in_h1_and_says_why(env):
    bad = ex("print(undefined_name)")
    st, rounds, trig, chat, sb = env([bad, bad, FIN, FIN, FIN], tier="H1")
    assert "Not run" in obs(rounds, 2) and "failed with" in obs(rounds, 2)
    st0, rounds0, *_ = env([bad, bad, FIN, FIN, FIN], tier="H0")
    assert "Not run" not in obs(rounds0, 2)


def test_timeout_restart_replays_all_blocks(env):
    st, rounds, trig, chat, sb = env([
        ex("a = 41"),
        ex("b = a + 1\nraise ValueError('boom')"),          # fails but leaves b
        ex("import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\ntime.sleep(60)"),  # ignores INT -> kill + restart
        ex("print(a, b)"), FIN, FIN, FIN], tier="H0", timeout=8)
    assert "re-run in order" in obs(rounds, 3)
    assert "41 42" in obs(rounds, 4)


def test_emergency_save_h1_only(env):
    mk = ex("res = gpd.read_file('dataset/pts.geojson')\nres['w'] = res['v'] * 2")
    st, rounds, trig, chat, sb = env([mk, ex("print(1)"), ex("print(2)")], tier="H1", max_rounds=3)
    assert os.path.exists(os.path.join(sb.work_dir, "pred_results", "emergency_output.geojson"))
    st0, rounds0, trig0, chat0, sb0 = env([mk, ex("print(1)"), ex("print(2)")], tier="H0", max_rounds=3)
    assert not os.path.exists(os.path.join(sb0.work_dir, "pred_results", "emergency_output.geojson"))


def test_observation_caps_geometry(env):
    code = ("from shapely.geometry import Polygon\nimport numpy as np\nt = np.linspace(0, 6.28, 200000)\n"
            "big = gpd.GeoDataFrame({'k': [1]}, geometry=[Polygon(zip(568000 + 1e4*np.cos(t), 300000 + 1e4*np.sin(t)))], crs=32616)\n"
            "print(big.head())")
    st, rounds, trig, chat, sb = env([ex(code), call("inspect", {"var_name": "big"}), FIN, FIN, FIN],
                                     tier="H1")
    assert len(obs(rounds, 1)) <= 6100 and len(obs(rounds, 2)) <= 4100
    assert "POLYGON" not in obs(rounds, 2)


def test_g1_card_in_prompt(env):
    st, rounds, trig, chat, sb = env([FIN, FIN, FIN], tier="H2")
    task_msg = chat.seen[0][1]["content"]
    assert "Data profile" in task_msg and "EPSG:32616" in task_msg and "30 features" in task_msg
    st0, r0, t0, chat0, _ = env([FIN, FIN, FIN], tier="H1")
    assert "Data profile" not in chat0.seen[0][1]["content"]


class KindChat(ScriptChat):
    """Replies chosen by call kind: planner / verifier lists, everything else the worker list."""

    def __init__(self, worker, planner, verifier):
        super().__init__(worker)
        self.planner, self.verifier = list(planner), list(verifier)

    def chat(self, messages, tools=None, kind="agent"):
        if kind == "planner":
            return Reply(self.planner.pop(0) if len(self.planner) > 1 else self.planner[0])
        if kind == "verifier":
            return Reply(self.verifier.pop(0) if len(self.verifier) > 1 else self.verifier[0])
        roles = [m["role"] for m in messages[1:]]
        assert all(a != b for a, b in zip(roles, roles[1:])), f"roles do not alternate: {roles}"
        return super().chat(messages, tools, kind)


def test_dual_agent_steps_verify_retry(env, tmp_path):
    import dual
    base = tempfile.mkdtemp(prefix="dualtest_", dir=TMP_ROOT)
    ds = os.path.join(base, "dataset"); os.makedirs(ds)
    with open(os.path.join(ds, "pts.geojson"), "w") as f:
        json.dump(pts_fc(), f)
    cell = os.path.join(base, "cell"); work = os.path.join(cell, "work"); os.makedirs(work)
    sb = ContainerSandbox(work, timeout=30, dataset_dir=ds, journal_replay=True, cell_id="dualtest")
    try:
        plan = json.dumps({"steps": [{"id": 1, "goal": "load points", "produces": "variable pts"},
                                     {"id": 2, "goal": "save output", "produces": "pred_results/out.geojson"}]})
        worker = [ex("pts = gpd.read_file('dataset/pts.geojson')"), fin("loaded"),
                  ex("print(len(pts))"), fin("loaded, 30"),
                  ex("pts.to_file('pred_results/out.geojson')"), FIN, FIN]
        verifier = ['{"pass": false, "reason": "count not printed", "fix_hint": "print len(pts)"}',
                    '{"pass": true, "reason": "ok", "fix_hint": ""}']
        chat = KindChat(worker, [plan], verifier)
        ag = dual.DualAgent(chat, sb, ds, cell, tier="H2", verbose=False)
        st = ag.run(TASK)
        assert st["finished"] and st["end_reason"] == "finish"
        assert st["planner_calls"] == 1 and st["verifier_calls"] == 2 and st["replans"] == 0
        assert st["steps"][0]["pass"] is True
        assert os.path.exists(os.path.join(work, "pred_results", "out.geojson"))
    finally:
        sb.close()


def kinds(trig):
    return [t["kind"] for t in trig]


def test_g6_lookup_and_suggest(env):
    st, rounds, trig, chat, sb = env([
        call("lookup_api", {"query": "local getis ord"}),
        ex("from esda.getisord import getisord_gi"),
        ex("import libpysal\nw = libpysal.weights.KNNN"),
        FIN, FIN, FIN], tier="H2")
    assert "G_Local" in obs(rounds, 1)
    assert "G_Local" in obs(rounds, 2) and "has no 'getisord_gi'" in obs(rounds, 2)
    assert "KNN" in obs(rounds, 3)
    assert kinds(trig).count("suggest") == 2
    st1, rounds1, *_ = env([call("lookup_api", {"query": "x"}), FIN, FIN, FIN], tier="H2-g6")
    assert any(r.get("format_error") == "unknown_tool" for r in rounds1)  # tool absent without G6


def test_g2_merge_reprojection_weights_interpolation(env):
    st, rounds, trig, chat, sb = env([
        ex("import pandas as pd\nleft = pd.DataFrame({'id': [f'{i:05d}' for i in range(40)], 'v': range(40)})\n"
           "right = pd.DataFrame({'id': [str(i) for i in range(40)], 'w': range(40)})\nm = left.merge(right, on='id', how='left')"),
        ex("bad = gpd.read_file('dataset/pts.geojson').set_crs(4326, allow_override=True).to_crs(3857)"),
        ex("import libpysal\np = gpd.read_file('dataset/pts.geojson')\nw = libpysal.weights.DistanceBand.from_dataframe(p, threshold=100, silence_warnings=True)"),
        ex("from scipy.interpolate import Rbf\nimport numpy as np\nr = Rbf([0, 1, 2, 3], [0, 1, 0, 1], [1., 2., 1., 2.], function='linear')\nz = r(np.array([50.]), np.array([50.]))"),
        ex("ok = left.merge(left, on='id')"),                                   # all keys match: no note
        FIN, FIN, FIN], tier="H2")
    k = kinds(trig)
    for want in ("merge_few_matches", "reprojection_invalid", "weights_islands", "interpolation_out_of_range"):
        assert want in k, want
    assert k.count("merge_few_matches") == 1


def test_g3_figure_misplaced_and_colorbar(env):
    st, rounds, trig, chat, sb = env([
        ex("p = gpd.read_file('dataset/pts.geojson')\nax = p.plot()\nplt.savefig('pred_results/ok.png'); plt.close('all')"),
        ex("q = p.copy(); q.geometry = q.translate(5e6, 5e6)\nax = q.plot()\nplt.savefig('pred_results/off.png'); plt.close('all')"),
        ex("import numpy as np\nfig, ax = plt.subplots(); ax.imshow(np.random.rand(40, 40), extent=(568000, 583000, 300000, 315000))\n"
           "fig.savefig('pred_results/nocbar.png'); plt.close('all')"),
        FIN, FIN, FIN], tier="H2")
    misplaced = [t for t in trig if t["kind"] == "figure_misplaced"]
    assert len(misplaced) == 1 and "off.png" in misplaced[0]["detail"]["msg"]
    assert any(t["kind"] == "figure_no_colorbar" and "nocbar.png" in t["detail"]["msg"] for t in trig)


def test_g7_provenance(env):
    st, rounds, trig, chat, sb = env([
        ex("a = gpd.read_file('dataset/pts.geojson')\nb = a[a['v'] > 3]"),
        ex("b.to_file('pred_results/out.geojson')"), FIN, FIN], tier="H2")
    cell = os.path.dirname(sb.work_dir)
    prov = json.load(open(os.path.join(cell, "provenance.json")))
    o = prov["outputs"]["out.geojson"]
    assert o["written_in_rounds"] == [2] and o["inputs_read_before"] == ["dataset/pts.geojson"] and o["code_rounds"] == [1, 2]


def test_claude_message_conversion_append_only():
    from claude_client import ClaudeChat
    raw1 = [{"type": "thinking", "thinking": "", "signature": "sig1"},
            {"type": "tool_use", "id": "tu1", "name": "execute", "input": {"code": "x=1"}},
            {"type": "tool_use", "id": "tu2", "name": "execute", "input": {"code": "y=2"}}]
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "TASK"},
            {"role": "assistant", "content": None, "_raw": raw1, "tool_calls": []},
            {"role": "tool", "tool_call_id": "tu1", "content": "ok"},
            {"role": "tool", "tool_call_id": "tu2", "content": "Not run: one tool call per reply."},
            {"role": "assistant", "content": "All files are saved.", "_raw": None},
            {"role": "user", "content": "Format error: call a tool."}]
    system, out = ClaudeChat.convert(msgs)
    assert system == "SYS" and out[0] == {"role": "user", "content": "TASK"}
    assert out[1]["content"] is raw1                                    # sent back unchanged, signature kept
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["tu1", "tu2"]  # one user message, every call answered
    assert out[3]["content"] == [{"type": "text", "text": "All files are saved."}]
    # the OpenAI-compatible client never sends harness keys
    from llm import ChatClient
    c = ChatClient.__new__(ChatClient); c.model, c.set, c.drop = "m", {}, set()
    assert all("_raw" not in m for m in c._params(msgs, None)["messages"])


def test_single_param_bare_value():
    # a model can send the code itself as the arguments of execute
    from protocol import parse_reply
    T = loop.ALL_TOOLS
    a = parse_reply("", [{"id": "c1", "name": "execute", "arguments": "import numpy as np\nprint(np.pi)"}], T)
    assert a.name == "execute" and a.args == {"code": "import numpy as np\nprint(np.pi)"}
    b = parse_reply("", [{"id": "c2", "name": "execute", "arguments": '{"code": "print(1)"}'}], T)
    assert b.args == {"code": "print(1)"}
    c = parse_reply("", [{"id": "c3", "name": "load_vector", "arguments": "dataset/a.shp"}], T)   # two params: still an error
    assert c.kind == "bad_args"
    d = parse_reply("", [{"id": "c4", "name": "execute", "arguments": '{"code": "print(1)"'}], T)  # broken JSON object: error
    assert d.kind == "bad_args"


def test_g3_projected_plot_ok_and_mixed_crs(env):
    # a correct map drawn in EPSG:3857 (not an input CRS) must not be flagged
    st, rounds, trig, chat, sb = env([
        ex("p = gpd.read_file('dataset/pts.geojson')\nm = p.to_crs(3857)\nax = m.plot()\nplt.savefig('pred_results/merc.png'); plt.close('all')"),
        ex("fig, ax = plt.subplots(); p.plot(ax=ax); p.to_crs(4326).plot(ax=ax)\nfig.savefig('pred_results/mixed.png'); plt.close('all')"),
        FIN, FIN, FIN], tier="H2")
    k = [(t["kind"], t["detail"].get("msg", "")) for t in trig]
    assert not any(kind == "figure_misplaced" for kind, _ in k)
    assert any(kind == "figure_mixed_crs" and "mixed.png" in m for kind, m in k)


def test_list_files_shows_outputs_and_readonly_repeat(env):
    LF = call("list_files", {})
    st, rounds, trig, chat, sb = env([ex("open('pred_results/a.txt', 'w').write('x')"), LF, LF, LF, FIN, FIN], tier="H1")
    assert "pred_results/a.txt" in obs(rounds, 2)
    assert obs(rounds, 3).startswith("Same result as round 2")
    st0, rounds0, *_ = env([LF, LF, FIN, FIN], tier="H0")   # H0 has no list_files tool at all
    assert any(r.get("format_error") == "unknown_tool" for r in rounds0)


def test_input_ablation_hides_dd_dk(env):
    import loop as L
    task = dict(TASK, dataset_description="DD-MARKER text", domain_knowledge="DK-MARKER text")
    for tier, dd, dk in (("H1", True, True), ("H1+nodd", False, True), ("H1+nodk", True, False)):
        base = tempfile.mkdtemp(prefix="abl_", dir=TMP_ROOT); w = os.path.join(base, "work"); os.makedirs(w)
        ds = os.path.join(base, "dataset"); os.makedirs(ds); open(os.path.join(ds, "a.csv"), "w").write("x\n1\n")
        sb = ContainerSandbox(w, timeout=30, dataset_dir=ds, journal_replay=True, cell_id="abl" + tier.replace("+", ""))
        try:
            chat = ScriptChat([FIN, FIN])
            L.Agent(chat, sb, ds, base, tier=tier, verbose=False).run(task)
            msg = chat.seen[0][1]["content"]
            assert ("DD-MARKER" in msg) is dd and ("DK-MARKER" in msg) is dk and "a.csv" in msg
        finally:
            sb.close()


def test_hint_tier_only_with_hint(env):
    import loop as L
    from prompts import task_message
    assert "hint" not in L.tier_config("H2") and L.tier_config("H2+hint")["hint"] is True
    base_t = dict(TASK, domain_knowledge="DK text")
    # an empty note leaves the task message unchanged
    assert task_message(base_t, "f") == task_message(dict(base_t, hint=""), "f")
    assert "## Additional Note\nHINT-MARKER" in task_message(dict(base_t, hint="HINT-MARKER"), "f")
    for tier, want in (("H1", False), ("H1+hint", True)):
        base = tempfile.mkdtemp(prefix="hint_", dir=TMP_ROOT); w = os.path.join(base, "work"); os.makedirs(w)
        ds = os.path.join(base, "dataset"); os.makedirs(ds); open(os.path.join(ds, "a.csv"), "w").write("x\n1\n")
        sb = ContainerSandbox(w, timeout=30, dataset_dir=ds, journal_replay=True, cell_id="hint" + tier.replace("+", ""))
        try:
            chat = ScriptChat([FIN, FIN])
            L.Agent(chat, sb, ds, base, tier=tier, verbose=False).run(dict(TASK, hint="HINT-MARKER"))
            assert ("HINT-MARKER" in chat.seen[0][1]["content"]) is want
        finally:
            sb.close()
