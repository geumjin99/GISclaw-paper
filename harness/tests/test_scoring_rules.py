"""Scoring rules: CRS of deliverable files, copies of inputs, replay faithfulness, scorer
crashes, code text of failed blocks. No container needed.

    python3 -m pytest -q harness/tests/test_scoring_rules.py
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import checklib  # noqa: E402
import nocrs  # noqa: E402

gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")
from shapely.geometry import Point  # noqa: E402

CONTRACT = {"task": 0, "deliverables": [
    {"id": "v", "kind": "vector", "patterns": ["*.shp", "*.geojson", "*.gpkg"], "required": True,
     "crs_required": True, "checked_by": ["x"]},
    {"id": "r", "kind": "raster", "patterns": ["*.tif"], "required": True, "crs_required": True, "checked_by": ["x"]}]}


def _pts(crs="EPSG:32611"):
    return gpd.GeoDataFrame({"a": [1, 2, 3]}, geometry=[Point(500000 + i, 4000000 + i) for i in range(3)], crs=crs)


def _tif(path, crs="EPSG:32611"):
    from rasterio.transform import from_origin
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32", crs=crs,
                       transform=from_origin(500000, 4000000, 10, 10)) as w:
        w.write(np.arange(16, dtype="float32").reshape(1, 4, 4))


@pytest.mark.parametrize("ext", ["shp", "geojson", "gpkg"])
def test_vector_with_and_without_crs(tmp_path, ext):
    p = str(tmp_path / f"out.{ext}")
    _pts().to_file(p)
    assert checklib.geo_file_crs(p)[1]
    assert checklib.deliverable_crs_check(str(tmp_path), CONTRACT)["passed"]
    nocrs.write_without_crs(p)
    assert checklib.geo_file_crs(p)[1] is None
    r = checklib.deliverable_crs_check(str(tmp_path), CONTRACT)
    assert not r["passed"] and r["required"] and "out." in r["note"]


def test_geojson_projected_coords_without_crs_member_fails(tmp_path):
    # a GeoDataFrame without CRS written to GeoJSON reads back as EPSG:4326 - the coordinates
    # tell that it is not lon/lat
    p = str(tmp_path / "out.geojson")
    _pts(crs=None).to_file(p, driver="GeoJSON")
    assert gpd.read_file(p).crs is not None          # the reader's default label
    assert checklib.geo_file_crs(p)[1] is None
    assert not checklib.deliverable_crs_check(str(tmp_path), CONTRACT)["passed"]


def test_geojson_lonlat_without_crs_member_passes(tmp_path):
    p = str(tmp_path / "out.geojson")
    _pts().to_crs(4326).to_file(p, driver="GeoJSON")
    assert checklib.geo_file_crs(p)[1]
    assert checklib.deliverable_crs_check(str(tmp_path), CONTRACT)["passed"]


def test_raster_without_crs(tmp_path):
    p = str(tmp_path / "out.tif")
    _tif(p)
    assert checklib.deliverable_crs_check(str(tmp_path), CONTRACT)["passed"]
    nocrs.write_without_crs(p)
    assert not checklib.deliverable_crs_check(str(tmp_path), CONTRACT)["passed"]


def test_deliverable_in_subfolder_checked(tmp_path):
    (tmp_path / "sub").mkdir()
    p = str(tmp_path / "sub" / "out.tif")
    _tif(p)
    assert checklib.deliverable_crs_check(str(tmp_path), CONTRACT)["passed"]
    nocrs.write_without_crs(p)
    r = checklib.deliverable_crs_check(str(tmp_path), CONTRACT)
    assert not r["passed"] and "sub/out.tif" in r["note"]


def test_non_deliverable_and_non_geo_files_ignored(tmp_path):
    _pts(crs=None).to_file(str(tmp_path / "scratch.gpkg"))
    c = {"task": 0, "deliverables": [{"id": "v", "kind": "vector", "patterns": ["result.gpkg"],
                                      "required": True, "crs_required": True, "checked_by": ["x"]}]}
    assert checklib.deliverable_crs_check(str(tmp_path), c)["passed"]
    json.dump({"a": 1}, open(tmp_path / "table.json", "w"))
    assert checklib.geo_file_crs(str(tmp_path / "table.json"))[0] is None
    assert checklib.deliverable_crs_check(str(tmp_path), {"task": 0, "deliverables": []}) is None


def test_input_copies(tmp_path):
    inp, pred = tmp_path / "in", tmp_path / "pred"
    inp.mkdir()
    pred.mkdir()
    g = _pts()
    g.to_file(str(inp / "layer.shp"))
    _tif(str(inp / "dem.tif"))
    import shutil
    for n in os.listdir(inp):
        if n.startswith("layer."):
            shutil.copy(inp / n, pred / ("copy" + n[5:]))
    g.to_crs(4326).to_file(str(pred / "resaved.geojson"), driver="GeoJSON")    # same features, re-saved
    _tif(str(pred / "dem_copy.tif"))
    g2 = g.copy()
    g2["b"] = [7, 8, 9]                                                     # new attribute: a result
    g2.to_file(str(pred / "result.gpkg"))
    checklib._INPUT_INDEX.clear()
    cp = checklib.input_copies(str(pred), str(inp))
    assert {"copy.shp", "copy.dbf", "copy.shx", "copy.prj", "resaved.geojson", "dem_copy.tif"} <= set(cp)
    assert "result.gpkg" not in cp


def test_replay_faithfulness():
    from score import replay_faithfulness
    entries = [{"success": True}, {"success": False}]
    ok = {"block_success": [True, False], "block_outcome": ["ok", "ok"], "events": []}
    assert replay_faithfulness(entries, ok)[0]
    cut = dict(ok, block_outcome=["ok", "timeout_interrupted"])
    assert not replay_faithfulness(entries, cut)[0]
    part = dict(ok, events=[{"event": "replay_partial"}])
    assert not replay_faithfulness(entries, part)[0]
    diff = dict(ok, block_success=[False, False])
    assert not replay_faithfulness(entries, diff)[0]


def test_scorer_crash_is_not_a_verdict(tmp_path):
    ok = checklib.result("a", True)
    crash = checklib.scorer_error_result("Traceback ...")
    assert crash["scorer_error"] and crash["name"] == checklib.SCORER_ERROR_NAME
    assert checklib.verdict([ok]) is True
    assert checklib.verdict([ok, checklib.result("b", False)]) is False
    assert checklib.verdict([ok, crash]) is None
    assert checklib.verdict([]) is False                 # no required check: not correct


def test_point_set_distance_ignores_missing_geometries():
    a = gpd.GeoDataFrame(geometry=[Point(0, 0), None, Point()], crs="EPSG:32611")
    b = gpd.GeoDataFrame(geometry=[Point(3, 4)], crs="EPSG:32611")
    assert checklib.point_set_distance(a, b) == pytest.approx(5.0)
    empty = gpd.GeoDataFrame(geometry=[None, Point()], crs="EPSG:32611")
    assert np.isnan(checklib.point_set_distance(empty, b))


def test_failed_block_prefix():
    import score as RE
    code = "\nimport x\n\na = 1\nfor i in range(3):\n    b = i\nc = undefined\nd = 2\n"
    obs = ('stderr:\nTraceback (most recent call last):\n  File "/opt/k/sandbox.py", line 157, in execute\n'
           '    exec(code, self.namespace)\n  File "<string>", line 7, in <module>\nNameError: c')
    pre = RE.failed_block_prefix(code, obs)
    assert "a = 1" in pre and "b = i" in pre and "undefined" not in pre and "d = 2" not in pre
    # error inside a compound statement: the whole statement is left out
    obs6 = obs.replace("line 7, in <module>", "line 6, in <module>")
    pre6 = RE.failed_block_prefix(code, obs6)
    assert "a = 1" in pre6 and "for i" not in pre6
    # no locatable line (OOM, blocked import, syntax error, truncated): nothing is used
    assert RE.failed_block_prefix(code, "container out of memory") is None
    assert RE.failed_block_prefix("a = (", 'Traceback (most recent call last):\n  File "<string>", line 1\n') is None


def test_replay_container_failure():
    import score as RE
    entries = [{"outcome": "ok"}, {"outcome": "oom"}, {"outcome": "ok"}]
    # same kill as the original (oom / crash are the same event): no failure
    assert RE.replay_container_failure(entries, {"block_outcome": ["ok", "crash", "ok"], "events": []}, []) == []
    # a kill the original did not have
    assert RE.replay_container_failure(entries, {"block_outcome": ["oom", "ok", "ok"], "events": []}, [])
    # partial restore only in the replay
    run = {"block_outcome": ["ok", "oom", "ok"], "events": [{"event": "replay_partial", "failed_blocks": [0]}]}
    assert RE.replay_container_failure(entries, run, [{"event": "restarted"}])
    assert RE.replay_container_failure(entries, run, [{"event": "replay_partial"}]) == []
