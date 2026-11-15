"""Task-check helpers must not crash on non-finite coordinates (NaN offsets, missing / empty
geometries): scipy's cKDTree and spatial joins reject them. No container needed.

    python3 -m pytest -q harness/tests/test_nan_guards.py
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402
import checklib  # noqa: E402

gpd = pytest.importorskip("geopandas")
from shapely.geometry import Point, Polygon  # noqa: E402


def spec(t):
    p = os.path.join(paths.SPECS, str(t), "checks.py")
    s = importlib.util.spec_from_file_location(f"checks_guard_{t}", p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def snap_ctx(tmp_path, arrays):
    """Ctx whose pred snapshot holds figure arrays {name: (artist, array)} in tag save00."""
    d = tmp_path / "snap" / "save00"
    d.mkdir(parents=True)
    idx = []
    for name, (artist, a) in arrays.items():
        np.save(d / (name + ".npy"), np.asarray(a, float))
        idx.append({"name": name, "kind": "array", "shape": list(np.shape(a)), "dtype": "float64",
                    "source": "figure", "artist": artist})
    (d / "index.json").write_text(json.dumps(idx))
    (tmp_path / "pred").mkdir()
    return checklib.Ctx(pred_dir=str(tmp_path / "pred"), gold_dir=str(tmp_path), inputs_dir=str(tmp_path),
                        pred_snap=str(tmp_path / "snap"))


def _grid_pts(n=30, crs="EPSG:32611"):
    return gpd.GeoDataFrame(geometry=[Point(500000 + 10 * (i % 6), 4000000 + 10 * (i // 6)) for i in range(n)], crs=crs)


def test_t05_points_drawn_nan_offsets(tmp_path):
    cases = _grid_pts()
    off = np.c_[cases.geometry.x, cases.geometry.y]
    off[0] = np.nan
    ctx = snap_ctx(tmp_path, {"fig_ax0_coll0_offsets": ("PathCollection", off)})
    n, frac, where = spec(5)._points_drawn(ctx, cases, ["EPSG:32611"])
    assert n == len(cases) - 1          # the finite drawn points are still counted


def test_t05_gistar_missing_cells():
    g = gpd.GeoDataFrame(geometry=[Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)])
                                   for x in range(5) for y in range(5)], crs="EPSG:32611")
    g.loc[3, "geometry"] = None
    z = spec(5)._gistar_z(g, np.arange(25))
    assert len(z) == 25 and np.isnan(z[3]) and np.isfinite(np.delete(z, 3)).all()


def test_t08_drawn_share_empty_parts(tmp_path):
    sq = [Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]) for x in range(3)]
    layer = gpd.GeoDataFrame(geometry=sq + [Polygon()], crs="EPSG:32617")
    boxes = np.array([g.bounds for g in sq])
    ctx = snap_ctx(tmp_path, {"fig_ax0_coll0_paths_bbox": ("PatchCollection", boxes)})
    assert spec(8)._drawn_share(ctx, "save00", layer, []) == pytest.approx(1.0)


def test_t18_hexbin_nan_offsets(tmp_path):
    trees = _grid_pts(60)
    off = np.c_[np.linspace(500000, 500050, 12), np.linspace(4000000, 4000090, 12)]
    off[0] = np.nan
    ctx = snap_ctx(tmp_path, {"fig_ax0_coll0_offsets": ("PolyCollection", off),
                              "fig_ax0_coll0_values": ("PolyCollection", np.linspace(0, 1, 12))})
    ok, note, sense = spec(18)._hexbin_share(ctx, trees, np.zeros(len(trees), bool))
    assert ok in (True, False)


def test_t19_score_points_missing_geometries():
    df = _grid_pts(30)
    df["do"] = np.linspace(5, 9, 30)
    df.loc[0, "geometry"] = None
    df.loc[1, "geometry"] = Point()
    P = np.c_[df.geometry.x, df.geometry.y][2:]
    yard = {"idw": np.linspace(5, 9, 30)[2:]}
    best = spec(19)._score_points(df, P, yard)
    assert best is not None and best[0] == pytest.approx(1.0)
