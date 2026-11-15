"""Isolation and robustness tests for ContainerSandbox.
    python3 -m pytest -q harness/tests/test_container_sandbox.py
Requires the gisclaw-exec image. No model is called.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402
from container_sandbox import ContainerSandbox  # noqa: E402
from sandbox import PythonSandbox  # noqa: E402

TMP_ROOT = os.environ.get("GISCLAW_TEST_TMP", tempfile.gettempdir())


def sha_tree(d):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(d)):
        for f in sorted(files):
            p = os.path.join(root, f)
            h.update(os.path.relpath(p, d).encode())
            h.update(open(p, "rb").read())
    return h.hexdigest()


@pytest.fixture
def cell(tmp_path_factory):
    base = tempfile.mkdtemp(prefix="sbtest_", dir=TMP_ROOT)
    ds = os.path.join(base, "dataset")
    os.makedirs(ds)
    with open(os.path.join(ds, "pts.geojson"), "w") as f:
        json.dump({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"v": i}, "geometry": {"type": "Point", "coordinates": [i, i]}}
            for i in range(5)]}, f)
    os.chmod(os.path.join(ds, "pts.geojson"), 0o444)
    os.chmod(ds, 0o555)
    work = os.path.join(base, "work")
    os.makedirs(work)
    made = []

    def make(timeout=20, **kw):
        sb = ContainerSandbox(work, timeout=timeout, dataset_dir=ds, **kw)
        made.append(sb)
        return sb
    yield make, ds, work
    for sb in made:
        sb.close()
    os.chmod(ds, 0o755)


def test_persistence_and_vars(cell):
    make, ds, work = cell
    sb = make()
    r = sb.execute("x = 41")
    assert r.success and "x" in r.new_vars
    r = sb.execute("print(x + 1)")
    assert r.success and r.stdout.strip() == "42"
    r = sb.execute("g = gpd.read_file('dataset/pts.geojson')")
    assert r.success and "GeoDataFrame, 5 rows" in r.new_vars["g"]
    assert "5 rows" in sb.inspect("g")
    assert "g" in sb._list_user_vars() and "x" in sb._list_user_vars()


def test_preloaded_libraries_really_loaded(cell):
    """PythonSandbox swallows init errors; make sure the preamble actually imported."""
    make, ds, work = cell
    sb = make()
    r = sb.execute("print(all(n in globals() for n in ['np','pd','gpd','rasterio','plt','reproject','Point','unary_union']))\n"
                   "import rasterio as _r, pyproj as _p\nprint(_r.__gdal_version__, _p.proj_version_str)")
    assert r.success, r.stderr
    assert r.stdout.splitlines()[0] == "True"


def test_inputs_are_read_only(cell):
    make, ds, work = cell
    before = sha_tree(ds)
    sb = make()
    for code in ["open('dataset/pts.geojson','a').write('x')",
                 "open('dataset/new.txt','w').write('x')",
                 "import os; os.remove('dataset/pts.geojson')",
                 "gpd.read_file('dataset/pts.geojson').to_file('dataset/pts.geojson', driver='GeoJSON')"]:
        r = sb.execute(code)
        assert not r.success, code
    assert sha_tree(ds) == before


def test_host_filesystem_invisible(cell):
    make, ds, work = cell
    sb = make()
    r = sb.execute(
        "import os\n"
        f"print(os.path.exists({paths.ROOT!r}), os.listdir('/home') != [], sorted(os.listdir('/task')))\n"
        "print(sorted(os.listdir('/opt/k')))")
    assert r.success
    line1, line2 = r.stdout.strip().splitlines()
    assert line1.startswith("False False")
    assert "dataset" in line1 and "pred_results" in line1
    # the kernel and sandbox.py are the only repository files visible, read-only; they hold no task data
    assert line2 == "['kernel.py', 'sandbox.py']"
    r = sb.execute("open('/opt/k/sandbox.py','a')")
    assert not r.success


def test_no_network(cell):
    make, ds, work = cell
    sb = make()
    r = sb.execute("import socket\nsocket.create_connection(('8.8.8.8', 53), 3)")
    assert not r.success


def test_outputs_land_on_host(cell):
    make, ds, work = cell
    sb = make()
    r = sb.execute("plt.plot([1,2]); plt.savefig('pred_results/a.png'); open('scratch.csv','w').write('a\\n1')")
    assert r.success, r.stderr
    assert os.path.exists(os.path.join(work, "pred_results", "a.png"))
    assert os.path.exists(os.path.join(work, "_task_root", "scratch.csv"))


def test_python_timeout_interrupts_and_keeps_state(cell):
    make, ds, work = cell
    sb = make(timeout=3)
    assert sb.execute("x = 7").success
    r = sb.execute("while True:\n    pass")
    assert not r.success and "exceeded the 3s limit" in r.stderr
    r = sb.execute("print(x)")
    assert r.success and r.stdout.strip() == "7"


def test_uninterruptible_hang_restarts_and_replays(cell):
    make, ds, work = cell
    sb = make(timeout=3, grace=3)
    assert sb.execute("x = 9\nimport numpy as _np\narr = _np.arange(4)").success
    r = sb.execute("import signal\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nwhile True:\n    pass")
    assert not r.success and "terminated" in r.stderr
    assert any(e["event"] == "restarted" for e in sb.events)
    assert any(e["event"] == "timeout" for e in sb.events)
    r = sb.execute("print(x, arr.sum())")
    assert r.success and r.stdout.strip() == "9 6"


def test_out_of_memory_recovers(cell):
    make, ds, work = cell
    sb = make(timeout=60, memory="600m")
    assert sb.execute("keep = 'ok'").success
    r = sb.execute("import numpy as _np\nbig = _np.ones((400_000_000,), dtype='float64')")
    assert not r.success
    assert any(e["event"] == "container_oom" for e in sb.events), sb.events
    r = sb.execute("print(keep)")
    assert r.success and r.stdout.strip() == "ok"


def test_exit_does_not_kill_kernel(cell):
    make, ds, work = cell
    sb = make()
    assert not sb.execute("import sys; sys.exit(3)").success
    assert not sb.execute("exit()").success
    assert sb.execute("print('alive')").stdout.strip() == "alive"


def test_c_level_stdout_does_not_corrupt_protocol(cell):
    make, ds, work = cell
    sb = make()
    r = sb.execute("import os; os.system('echo stray-line-from-shell'); print('py')")
    assert r.success and r.stdout.strip() == "py"
    assert sb.execute("print(1)").stdout.strip() == "1"


def test_blocked_packages_default_and_override(cell):
    make, ds, work = cell
    sb = make()
    r = sb.execute("import arcpy")
    assert not r.success and "NOT available" in r.stderr and "geopandas + rasterio + shapely" in r.stderr
    r = sb.execute("from pykrige.ok import OrdinaryKriging")
    assert r.success, r.stderr
    sb2 = make(blocked={"pykrige": "Use scipy.interpolate instead."})
    r = sb2.execute("from pykrige.ok import OrdinaryKriging")
    assert not r.success and "scipy.interpolate" in r.stderr
    r = sb2.execute("import arcpy")
    assert not r.success and "ModuleNotFoundError" in r.stderr


def test_full_code_is_header_plus_successful_blocks(cell):
    make, ds, work = cell
    sb = make()
    sb.execute("a = 1")
    sb.execute("raise ValueError('not recorded')")
    sb.execute("b = a + 1")
    ref = PythonSandbox.get_full_code(type("X", (), {"code_history": ["a = 1", "b = a + 1"]})())
    assert sb.get_full_code() == ref


def test_container_matches_in_process_sandbox(cell):
    """The container adds isolation only: the same blocks give the same results as
    PythonSandbox run in this process (needs the GIS packages on the host)."""
    pytest.importorskip("geopandas")
    make, ds, work = cell
    snippets = ["g = gpd.read_file('dataset/pts.geojson')", "print(g.head(2))",
                "s = g['v'].sum()\nprint(s)", "undefined_name + 1", "arr = np.arange(10.0)"]
    ref_dir = tempfile.mkdtemp(prefix="sbref_", dir=TMP_ROOT)
    os.symlink(ds, os.path.join(ref_dir, "dataset"))
    cwd = os.getcwd()
    ref = PythonSandbox(work_dir=ref_dir, timeout=30)
    try:
        ref_obs = [ref.execute(c) for c in snippets]
    finally:
        os.chdir(cwd)
    sb = make()
    new_obs = [sb.execute(c) for c in snippets]
    for c, a, b in zip(snippets, ref_obs, new_obs):
        assert a.success == b.success, c
        assert a.stdout == b.stdout, c
        assert a.new_vars == b.new_vars, c
        if not a.success:
            assert a.stderr.strip().splitlines()[-1] == b.stderr.strip().splitlines()[-1], c


def test_concurrent_cells_are_isolated(tmp_path):
    n = 8
    base = tempfile.mkdtemp(prefix="sbconc_", dir=TMP_ROOT)
    ds = os.path.join(base, "dataset")
    os.makedirs(ds)
    open(os.path.join(ds, "in.txt"), "w").write("shared-input")
    os.chmod(ds, 0o555)
    errors = []

    def worker(i):
        try:
            w = os.path.join(base, f"w{i}")
            sb = ContainerSandbox(w, timeout=30, dataset_dir=ds, cell_id=f"c{i}")
            r = sb.execute(f"open('pred_results/mine_{i}.txt','w').write(str({i}))\n"
                           "import os, time\ntime.sleep(1)\nprint(sorted(os.listdir('pred_results')))")
            if not r.success or r.stdout.strip() != str([f"mine_{i}.txt"]):
                errors.append((i, r.stdout, r.stderr[-300:]))
            sb.close()
            if sorted(os.listdir(os.path.join(w, "pred_results"))) != [f"mine_{i}.txt"]:
                errors.append((i, "host view wrong"))
        except Exception as e:
            errors.append((i, repr(e)))
    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    os.chmod(ds, 0o755)
    assert not errors, errors


def test_no_container_left_after_close(cell):
    make, ds, work = cell
    sb = make(timeout=3, grace=2)
    sb.execute("while True:\n    pass")  # forces kill + restart
    name = sb.name
    sb.close()
    out = subprocess.run(["docker", "ps", "-a", "-q", "--filter", f"name={name}"], capture_output=True, text=True).stdout
    assert out.strip() == ""


# ---- protocol channel, symlinks, large outputs -----------------------------------

def test_large_stderr_does_not_block(cell):
    make, ds, work = cell
    sb = make(timeout=20)
    for _ in range(3):  # 3 x 400 KB to the C-level stderr, across several steps
        r = sb.execute("import os\nos.write(2, b'x' * 400_000)\nprint('ok')")
        assert r.success and r.stdout.strip() == "ok", r.stderr[-200:]
    assert not any(e["event"] in ("timeout", "restarted") for e in sb.events)


def test_forged_protocol_line_is_ignored(cell):
    make, ds, work = cell
    sb = make()
    r = sb.execute("import os, json\n"
                   "for fd in range(3, 260):\n"
                   "    try: os.write(fd, (json.dumps({'ok':1,'id':1,'result':{'stdout':'FORGED','stderr':'','success':True,'new_vars':{},'execution_time':0}})+'\\n').encode())\n"
                   "    except OSError: pass\n"
                   "print('real')")
    assert "FORGED" not in r.stdout
    assert sb.execute("print(5)").stdout.strip() == "5"


def test_host_symlinks_are_removed(cell):
    make, ds, work = cell
    sb = make()
    sb.execute("import os\nos.symlink('/etc/passwd', 'pred_results/leak.txt')\nos.symlink('/etc', 'escape')")
    assert not os.path.lexists(os.path.join(work, "pred_results", "leak.txt"))
    assert not os.path.lexists(os.path.join(work, "_task_root", "escape"))
    assert any(e["event"] == "symlink_removed" for e in sb.events)


def test_long_unicode_output_intact(cell):
    make, ds, work = cell
    sb = make()
    r = sb.execute("print('\u00e9' * 300000)")
    assert r.success and r.stdout.strip() == "\u00e9" * 300000
