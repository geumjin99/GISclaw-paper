"""Run Python code in the isolated container and capture the data behind its outputs.

Many tasks only ask for a map (PNG). To judge whether the *analysis* is right we capture,
at every figure save and once at the end, the GeoDataFrames / DataFrames / numeric arrays /
raster metadata that live in the namespace. The same capture is applied to reference
solutions (gold) and, later, to the agents' code (replayed), so both sides are comparable.

Snapshots land in <work_dir>/_task_root/_snap/<tag>/  (tag = save00, save01, ..., final)
with an index.json describing every captured object.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fixed random state for the replay, so that a run gets the same verdict every time it is
# scored: code that draws random numbers without a seed (permutation tests such as LISA / Gi*,
# random forests, simulation envelopes) otherwise gives a different result on each replay.
# Seeds Python, NumPy's global generator and numba's own generator (used by esda's parallel
# permutation code). The seed is the same for every run.
SEED_PRELUDE = r'''
import random as _cap_random, numpy as _cap_np
_cap_random.seed(0)
_cap_np.random.seed(0)
try:
    import numba as _cap_numba

    @_cap_numba.njit(cache=False)
    def _cap_numba_seed(s):
        _cap_np.random.seed(s)
    _cap_numba_seed(0)
    del _cap_numba_seed
except Exception:
    pass
del _cap_random, _cap_np
'''

PRELUDE = r'''
def _cap_install_capture():
    import os, json, numpy as _np, pandas as _pd, matplotlib.figure as _mf
    try:
        import geopandas as _gpd
    except Exception:
        _gpd = None
    _MAX_BYTES = 1536 * 1024 * 1024
    _TOTAL_CAP = 3 * 1024 ** 3   # savefig snapshots stop beyond this (final snapshot always taken)
    _RET_CAP = 1 * 1024 ** 3     # function-return snapshots stop beyond this, so they cannot starve savefig
    _state = {"n": 0, "r": 0}

    def _used(prefix=None):
        tot = 0
        for r, _, fs in os.walk("/task/_snap"):
            if prefix and not os.path.basename(r).startswith(prefix):
                continue
            for f in fs:
                try:
                    tot += os.path.getsize(os.path.join(r, f))
                except OSError:
                    pass
        return tot
    _g = globals()

    def _snap(tag, extra=None, fig=None, with_globals=True):
        d = os.path.join("/task/_snap", tag)
        os.makedirs(d, exist_ok=True)
        idx = []
        over = (tag.startswith("ret") and _used("ret") > _RET_CAP) or \
               (tag.startswith("save") and _used() > _TOTAL_CAP)
        if over:
            json.dump([{"name": "_cap", "kind": "error", "error": "snapshot budget exhausted"}],
                      open(os.path.join(d, "index.json"), "w"))
            return
        items = dict(_g) if with_globals else {}
        items.update(extra or {})  # locals of user functions on the call stack win
        for k, v in list(items.items()):
            if k.startswith("_") or k in ("In", "Out"):
                continue
            try:
                if _gpd is not None and isinstance(v, _gpd.GeoDataFrame):
                    if v.memory_usage(deep=True).sum() > _MAX_BYTES:
                        idx.append({"name": k, "kind": "gdf", "skipped": "too large", "rows": len(v)}); continue
                    vv = v.copy()
                    gname = str(v.geometry.name)
                    vv.columns = [str(c) for c in vv.columns]
                    vv = vv.set_geometry(gname)
                    vv.to_parquet(os.path.join(d, k + ".parquet"))
                    idx.append({"name": k, "kind": "gdf", "rows": len(v), "crs": str(v.crs),
                                "columns": [str(c) for c in v.columns]})
                elif _gpd is not None and isinstance(v, _gpd.GeoSeries):
                    if len(v) == 0:
                        continue
                    _gpd.GeoDataFrame(geometry=v.reset_index(drop=True), crs=v.crs).to_parquet(
                        os.path.join(d, k + ".parquet"))
                    idx.append({"name": k, "kind": "gdf", "rows": len(v), "crs": str(v.crs),
                                "columns": ["geometry"], "source": "geoseries"})
                elif type(v).__module__.startswith("shapely") and hasattr(v, "geom_type") and hasattr(v, "is_empty"):
                    if v.is_empty:
                        continue
                    _gpd.GeoDataFrame(geometry=[v]).to_parquet(os.path.join(d, k + ".parquet"))
                    idx.append({"name": k, "kind": "gdf", "rows": 1, "crs": "None", "columns": ["geometry"],
                                "source": "shapely", "geom_type": v.geom_type})
                elif isinstance(v, _pd.DataFrame):
                    if v.memory_usage(deep=True).sum() > _MAX_BYTES:
                        idx.append({"name": k, "kind": "df", "skipped": "too large", "rows": len(v)}); continue
                    vv = v.copy()
                    vv.columns = [str(c) for c in vv.columns]
                    vv.reset_index().to_parquet(os.path.join(d, k + ".parquet"))
                    idx.append({"name": k, "kind": "df", "rows": len(v), "columns": [str(c) for c in v.columns]})
                elif isinstance(v, _np.ma.MaskedArray) and v.dtype.kind in "fiub" and v.size >= 4:
                    a = _np.ma.filled(v.astype(float), _np.nan)
                    if a.nbytes > _MAX_BYTES:
                        idx.append({"name": k, "kind": "array", "skipped": "too large", "shape": list(a.shape)}); continue
                    _np.save(os.path.join(d, k + ".npy"), a)
                    idx.append({"name": k, "kind": "array", "shape": list(a.shape), "dtype": "float64", "masked": True})
                elif type(v).__module__.startswith("xarray") and type(v).__name__ in ("DataArray", "Dataset"):
                    arrs = {k: v} if type(v).__name__ == "DataArray" else {f"{k}.{n}": v[n] for n in v.data_vars}
                    for nm, da in arrs.items():
                        a = _np.asarray(da.values)
                        if a.dtype.kind not in "fiub" or a.size < 4 or a.nbytes > _MAX_BYTES:
                            continue
                        _np.save(os.path.join(d, nm + ".npy"), a.astype(float) if a.dtype.kind == "b" else a)
                        idx.append({"name": nm, "kind": "array", "shape": list(a.shape), "dtype": str(a.dtype),
                                    "source": "xarray", "dims": list(da.dims)})
                elif type(v).__name__ == "Cube" and type(v).__module__.startswith("iris"):
                    a = _np.ma.filled(_np.ma.asarray(v.data).astype(float), _np.nan)
                    if a.size >= 4 and a.nbytes <= _MAX_BYTES:
                        _np.save(os.path.join(d, k + ".npy"), a)
                        idx.append({"name": k, "kind": "array", "shape": list(a.shape), "dtype": "float64",
                                    "source": "iris", "cube_name": str(v.name())})
                elif type(v).__module__.startswith("networkx") and hasattr(v, "nodes") and hasattr(v, "edges"):
                    if v.number_of_nodes() > 2_000_000:
                        continue
                    nd = _pd.DataFrame([{"node": str(n), **{a: b for a, b in attrs.items()
                                                            if isinstance(b, (int, float)) and not isinstance(b, bool)}}
                                        for n, attrs in v.nodes(data=True)])
                    nd.to_parquet(os.path.join(d, k + ".nodes.parquet"))
                    idx.append({"name": k + ".nodes", "kind": "df", "rows": len(nd), "columns": list(nd.columns),
                                "source": "networkx"})
                    ed = _pd.DataFrame([{"u": str(e[0]), "v": str(e[1]), **{a: b for a, b in e[-1].items()
                                                                          if isinstance(b, (int, float)) and not isinstance(b, bool)}}
                                        for e in v.edges(data=True)])
                    if len(ed):
                        ed.to_parquet(os.path.join(d, k + ".edges.parquet"))
                        idx.append({"name": k + ".edges", "kind": "df", "rows": len(ed), "columns": list(ed.columns),
                                    "source": "networkx"})
                elif isinstance(v, dict) and len(v) >= 4 and all(isinstance(x, (int, float, _np.integer, _np.floating))
                                                                 and not isinstance(x, bool) for x in v.values()):
                    sr = _pd.Series({str(a): float(b) for a, b in v.items()}, name=k)
                    sr.rename_axis("index").reset_index().to_parquet(os.path.join(d, k + ".parquet"))
                    idx.append({"name": k, "kind": "series", "rows": len(sr), "source": "dict"})
                elif type(v).__module__.split(".")[0] in ("esda", "mgwr", "pointpats", "spreg", "sklearn",
                                                          "pykrige", "statsmodels", "libpysal") and not isinstance(v, type):
                    # analysis result objects: keep their numeric array/scalar attributes
                    got = 0
                    for an in sorted(set(list(getattr(v, "__dict__", {}).keys()))):
                        if an.startswith("_") or got >= 40:
                            continue
                        try:
                            av = getattr(v, an)
                        except Exception:
                            continue
                        if isinstance(av, _np.ndarray) and av.dtype.kind in "fiub" and 1 <= av.size and av.nbytes <= 200e6:
                            _np.save(os.path.join(d, f"{k}.{an}.npy"), av)
                            idx.append({"name": f"{k}.{an}", "kind": "array", "shape": list(av.shape),
                                        "dtype": str(av.dtype), "source": "object_attr", "object_type": type(v).__name__})
                            got += 1
                        elif isinstance(av, (int, float, _np.integer, _np.floating)) and not isinstance(av, bool):
                            idx.append({"name": f"{k}.{an}", "kind": "scalar", "value": float(av),
                                        "source": "object_attr", "object_type": type(v).__name__})
                            got += 1
                    idx.append({"name": k, "kind": "object", "object_type": type(v).__module__ + "." + type(v).__name__})
                elif isinstance(v, _pd.Series) and v.dtype.kind in "fiub":
                    v.to_frame(name=str(v.name or k)).reset_index().to_parquet(os.path.join(d, k + ".parquet"))
                    idx.append({"name": k, "kind": "series", "rows": len(v)})
                elif isinstance(v, _np.ndarray) and v.dtype.kind in "fiub" and v.size >= 4:
                    if v.nbytes > _MAX_BYTES:
                        idx.append({"name": k, "kind": "array", "skipped": "too large", "shape": list(v.shape)}); continue
                    _np.save(os.path.join(d, k + ".npy"), v)
                    idx.append({"name": k, "kind": "array", "shape": list(v.shape), "dtype": str(v.dtype)})
                elif isinstance(v, (list, tuple)) and 4 <= len(v) <= 5_000_000 and all(
                        isinstance(x, (int, float, _np.integer, _np.floating)) and not isinstance(x, bool) for x in v[:1000]):
                    try:
                        a = _np.asarray(v, dtype=float)
                        if a.ndim == 1:
                            _np.save(os.path.join(d, k + ".npy"), a)
                            idx.append({"name": k, "kind": "array", "shape": list(a.shape), "dtype": "float64",
                                        "source": "list"})
                    except Exception:
                        pass
                elif isinstance(v, (int, float)) and not isinstance(v, bool):
                    idx.append({"name": k, "kind": "scalar", "value": float(v)})
                elif isinstance(v, dict) and "transform" in v and "crs" in v:
                    t = v.get("transform")
                    idx.append({"name": k, "kind": "raster_meta", "crs": str(v.get("crs")),
                                "transform": list(t)[:6] if t is not None else None,
                                "width": v.get("width"), "height": v.get("height"), "nodata": v.get("nodata")})
            except Exception as e:
                idx.append({"name": k, "kind": "error", "error": repr(e)[:200]})
        if fig is not None:
            idx.extend(_figure_data(fig, d))
            _drawn_layers(fig, d)
            _drawn_patches(fig, d)
        json.dump(idx, open(os.path.join(d, "index.json"), "w"), indent=0, default=str)

    # Geo layers drawn with GeoDataFrame.plot / GeoSeries.plot: single-colour layers leave no
    # data values in the figure capture, so for each such call we record the drawn object's
    # row count, CRS, total bounds and per-feature bounds, per figure. Written as
    # <snap>/drawn_layers.json (+ drawn_layer<j>_bounds.npy), NOT in index.json, so checks that
    # iterate the captured objects are unaffected.
    import weakref as _wr
    _drawn = _wr.WeakKeyDictionary()
    _busy = [False]

    def _root_fig(ax):
        f = getattr(ax, "figure", None)
        while f is not None and getattr(f, "figure", None) is not None and f.figure is not f:
            f = f.figure
        return f

    def _record_draw(obj, res, kw):
        try:
            ax = res if hasattr(res, "figure") else kw.get("ax")
            if ax is None:
                return
            fig = _root_fig(ax)
            if fig is None:
                return
            lst = _drawn.setdefault(fig, [])
            if len(lst) >= 60:
                return
            gs = obj.geometry if isinstance(obj, _gpd.GeoDataFrame) else obj
            rec = {"rows": int(len(obj)), "crs": str(obj.crs) if obj.crs is not None else None,
                   "type": type(obj).__name__,
                   "geom_types": {str(a): int(b) for a, b in gs.geom_type.value_counts().items()}}
            if len(obj):
                rec["total_bounds"] = [float(x) for x in gs.total_bounds]
            if 0 < len(obj) <= 300000:
                rec["_bounds"] = _np.asarray(gs.bounds.to_numpy(), float)
            lst.append(rec)
        except Exception:
            pass

    def _drawn_layers(fig, d):
        try:
            lst = _drawn.get(fig) or []
            if not lst:
                return
            out = []
            for j, rec in enumerate(lst):
                r = {k: v for k, v in rec.items() if k != "_bounds"}
                if rec.get("_bounds") is not None:
                    nm = "drawn_layer%02d_bounds.npy" % j
                    _np.save(os.path.join(d, nm), rec["_bounds"])
                    r["bounds_file"] = nm
                r["i"] = j
                out.append(r)
            json.dump(out, open(os.path.join(d, "drawn_layers.json"), "w"), default=str)
        except Exception:
            pass

    # Hand-drawn patches: ax.add_patch(Polygon / PathPatch / ...) one feature at a time leaves
    # no collection for the figure capture. Per axes, consecutive patches with the same look
    # (class, zorder, edge colour, line width, filled) form a group; per group we keep the face
    # colours (n x 4, alpha applied), the
    # vertices in data coordinates (cols: patch index, x, y) and per-patch bounds. Written as
    # <snap>/drawn_patches.json + drawn_patches_g<j>_*.npy, NOT in index.json, so checks that
    # iterate the captured objects are unaffected (read with capture.drawn_patch_groups).
    def _drawn_patches(fig, d):
        try:
            out, nv_tot, j = [], 0, 0
            for i, ax in enumerate(fig.axes):
                groups, cur, key0 = [], [], None
                for p in list(ax.patches)[:50000]:
                    try:
                        ec = tuple(_np.round(_np.asarray(p.get_edgecolor(), float), 3))
                        k = (type(p).__name__, float(p.get_zorder()), ec, round(float(p.get_linewidth() or 0), 3),
                             bool(p.get_fill()))
                    except Exception:
                        continue
                    if cur and k != key0:
                        groups.append((key0, cur)); cur = []
                    key0 = k
                    cur.append(p)
                if cur:
                    groups.append((key0, cur))
                for k, ps in groups:
                    fcs, verts, bbs = [], [], []
                    for q, p in enumerate(ps):
                        try:
                            v = p.get_patch_transform().transform(p.get_path().vertices)
                        except Exception:
                            v = _np.zeros((0, 2))
                        fcs.append(_np.asarray(p.get_facecolor(), float)[:4])
                        if len(v):
                            bbs.append([v[:, 0].min(), v[:, 1].min(), v[:, 0].max(), v[:, 1].max()])
                        else:
                            bbs.append([_np.nan] * 4)
                        if nv_tot + len(v) <= 500000:
                            verts.append(_np.c_[_np.full(len(v), q), v]); nv_tot += len(v)
                    rec = {"ax": i, "group": j, "n": len(ps), "type": k[0], "zorder": k[1],
                           "edgecolor": list(k[2]), "linewidth": k[3], "fill": k[4]}
                    _np.save(os.path.join(d, "drawn_patches_g%03d_facecolors.npy" % j), _np.asarray(fcs, float))
                    _np.save(os.path.join(d, "drawn_patches_g%03d_bbox.npy" % j), _np.asarray(bbs, float))
                    rec["facecolors"] = "drawn_patches_g%03d_facecolors.npy" % j
                    rec["bbox"] = "drawn_patches_g%03d_bbox.npy" % j
                    if verts and sum(len(x) for x in verts):
                        _np.save(os.path.join(d, "drawn_patches_g%03d_vertices.npy" % j), _np.concatenate(verts))
                        rec["vertices"] = "drawn_patches_g%03d_vertices.npy" % j
                        rec["vertices_complete"] = len(verts) == len(ps)
                    out.append(rec)
                    j += 1
            if out:
                json.dump(out, open(os.path.join(d, "drawn_patches.json"), "w"), default=str)
        except Exception:
            pass

    if _gpd is not None:
        try:
            import geopandas.plotting as _gplt

            def _wrap(orig, get_obj):
                def w(self, *a, **kw):
                    if _busy[0]:
                        return orig(self, *a, **kw)
                    _busy[0] = True
                    try:
                        res = orig(self, *a, **kw)
                    finally:
                        _busy[0] = False
                    kind = kw.get("kind", "geo")
                    if kind == "geo":
                        _record_draw(get_obj(self), res, kw)
                    return res
                w.__wrapped__ = orig
                w.__doc__ = getattr(orig, "__doc__", None)
                return w
            _gplt.GeoplotAccessor.__call__ = _wrap(_gplt.GeoplotAccessor.__call__, lambda s: s._parent)
            import inspect as _insp
            if _insp.isfunction(_gpd.GeoSeries.__dict__.get("plot")):  # else an accessor: covered above
                _gpd.GeoSeries.plot = _wrap(_gpd.GeoSeries.plot, lambda s: s)
        except Exception:
            pass

    def _figure_data(fig, d):
        """Data actually drawn in the figure: line x/y, image/mesh values, scatter offsets
        and colour values. Names: fig_ax<i>_<artist><j>[...]."""
        out = []
        def save(nm, a, what):
            a = _np.ma.filled(_np.ma.asarray(a, dtype=float), _np.nan)
            if a.size >= 4 and a.nbytes <= _MAX_BYTES:
                _np.save(os.path.join(d, nm + ".npy"), a)
                out.append({"name": nm, "kind": "array", "shape": list(a.shape), "dtype": "float64",
                            "source": "figure", "artist": what})
        try:  # texts shown in the figure (titles, axis labels, legend entries)
            texts = {"suptitle": fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else ""}
            for i, ax in enumerate(fig.axes):
                leg = ax.get_legend()
                texts[f"ax{i}"] = {"title": ax.get_title(), "xlabel": ax.get_xlabel(), "ylabel": ax.get_ylabel(),
                                   "legend": [t.get_text() for t in leg.get_texts()] if leg else []}
            out.append({"name": "fig_texts", "kind": "text", "source": "figure", "texts": texts})
        except Exception:
            pass
        for i, ax in enumerate(fig.axes):
            try:
                save(f"fig_ax{i}_limits", list(ax.get_xlim()) + list(ax.get_ylim()), "axes_limits")
                for j, ln in enumerate(ax.get_lines()):
                    save(f"fig_ax{i}_line{j}", ln.get_xydata(), "line")
                for j, im in enumerate(ax.get_images()):
                    save(f"fig_ax{i}_image{j}", im.get_array(), "image")
                    if out and out[-1]["name"] == f"fig_ax{i}_image{j}":
                        out[-1]["origin"] = getattr(im, "origin", None)
                        try:  # colour range actually used (vmin/vmax)
                            out[-1]["clim"] = [float(x) for x in im.get_clim()]
                        except Exception:
                            pass
                    save(f"fig_ax{i}_image{j}_extent", list(im.get_extent()), "image_extent")
                rects = [pt for pt in ax.patches if type(pt).__name__ == "Rectangle"]
                if len(rects) >= 2:  # bar charts: x, y, width, height of each bar
                    save(f"fig_ax{i}_bars", [[r.get_x(), r.get_y(), r.get_width(), r.get_height()] for r in rects],
                         "bars")
                for j, c in enumerate(ax.collections):
                    arr = c.get_array() if hasattr(c, "get_array") else None
                    if arr is not None:
                        save(f"fig_ax{i}_coll{j}_values", arr, type(c).__name__)
                        if out and out[-1]["name"] == f"fig_ax{i}_coll{j}_values":
                            try:
                                out[-1]["clim"] = [float(x) for x in c.get_clim()]
                            except Exception:
                                pass
                    try:
                        paths = c.get_paths() if hasattr(c, "get_paths") else []
                        if 1 <= len(paths) <= 500000:
                            bb = _np.array([[p.vertices[:, 0].min(), p.vertices[:, 1].min(),
                                             p.vertices[:, 0].max(), p.vertices[:, 1].max()]
                                            if len(p.vertices) else [_np.nan] * 4 for p in paths], float)
                            if bb.size >= 4:
                                save(f"fig_ax{i}_coll{j}_paths_bbox", bb, type(c).__name__)
                    except Exception:
                        pass
                    try:
                        if 1 <= len(paths) and sum(len(p.vertices) for p in paths) <= 200000:
                            vv = _np.concatenate([_np.c_[_np.full(len(p.vertices), q), p.vertices]
                                                  for q, p in enumerate(paths) if len(p.vertices)])
                            save(f"fig_ax{i}_coll{j}_vertices", vv, type(c).__name__)  # cols: path index, x, y
                    except Exception:
                        pass
                    if type(c).__name__ == "QuadMesh":
                        try:
                            co = c.get_coordinates()  # (rows+1, cols+1, 2) cell corners, drawn row order
                            if co.nbytes <= 200e6:
                                save(f"fig_ax{i}_coll{j}_mesh_coords", co, "QuadMesh")
                        except Exception:
                            pass
                    ec = c.get_edgecolors() if hasattr(c, "get_edgecolors") else None
                    if ec is not None and len(ec) >= 2:
                        save(f"fig_ax{i}_coll{j}_edgecolors", ec, type(c).__name__)
                    fc = c.get_facecolors() if hasattr(c, "get_facecolors") else None
                    if fc is not None and len(fc) >= 2:
                        save(f"fig_ax{i}_coll{j}_facecolors", fc, type(c).__name__)
                    off = c.get_offsets() if hasattr(c, "get_offsets") else None
                    if off is not None and len(off) >= 2:
                        save(f"fig_ax{i}_coll{j}_offsets", off, type(c).__name__)
            except Exception as e:
                out.append({"name": f"fig_ax{i}", "kind": "error", "error": repr(e)[:200]})
        return out

    _orig = _mf.Figure.savefig
    def _user_locals():
        """Locals of user-code frames (functions defined in the executed code) on the stack."""
        import sys as _sys
        out = {}
        f = _sys._getframe(2)
        while f is not None:
            if f.f_globals is _g and f.f_code.co_name != "<module>":
                for k, v in f.f_locals.items():
                    out.setdefault(k, v)
            f = f.f_back
        return out

    def _savefig(self, fname, *a, **kw):
        # which file each savefig wrote, in order (also beyond the 20 data snapshots), so a
        # check can look at the figure that ended up in the required file - not at another
        # figure, or at one that was later overwritten
        try:
            nm = os.path.basename(os.fspath(fname)) if isinstance(fname, (str, os.PathLike)) else None
            if nm and kw.get("format") and "." not in nm:
                nm = nm + "." + kw["format"]
            if nm and "." not in nm:
                nm = nm + ".png"
            _state.setdefault("saves", []).append(
                {"i": len(_state.get("saves", [])), "fname": nm,
                 "tag": ("save%02d" % _state["n"]) if _state["n"] < 20 else None})
            os.makedirs("/task/_snap", exist_ok=True)
            json.dump(_state["saves"], open("/task/_snap/savefig.json", "w"))
        except Exception:
            pass
        if _state["n"] < 20:
            try:
                _snap("save%02d" % _state["n"], _user_locals(), fig=self)
            except Exception:
                pass
            _state["n"] += 1
        # Agg's print_png writes through matplotlib.image.imsave: not a user imsave call
        _state["in_savefig"] = _state.get("in_savefig", 0) + 1
        try:
            return _orig(self, fname, *a, **kw)
        finally:
            _state["in_savefig"] -= 1
    _mf.Figure.savefig = _savefig

    # plt.imsave / matplotlib.image.imsave write an array straight to an image file, without a
    # Figure: record the call like a savefig (savefig.json + data snapshot) and, for a 2-D
    # numeric array, store it as the
    # drawn image fig_ax0_image0 (source "figure", artist "image", origin as drawn) so checks
    # on drawn images see it. RGB(A) arrays are colours, not data: only the save is recorded.
    import matplotlib.image as _mimg
    _orig_imsave = _mimg.imsave

    def _imsave(fname, arr, *a, **kw):
        if _state.get("in_savefig"):   # called by Figure.savefig (Agg print_png), not by user code
            return _orig_imsave(fname, arr, *a, **kw)
        try:
            nm = os.path.basename(os.fspath(fname)) if isinstance(fname, (str, os.PathLike)) else None
            if nm and kw.get("format") and "." not in nm:
                nm = nm + "." + kw["format"]
            if nm and "." not in nm:
                nm = nm + ".png"
            tag = ("save%02d" % _state["n"]) if _state["n"] < 20 else None
            _state.setdefault("saves", []).append({"i": len(_state.get("saves", [])), "fname": nm, "tag": tag,
                                                   "via": "imsave"})
            os.makedirs("/task/_snap", exist_ok=True)
            json.dump(_state["saves"], open("/task/_snap/savefig.json", "w"))
            if tag is not None:
                _state["n"] += 1
                _snap(tag, _user_locals())
                d = os.path.join("/task/_snap", tag)
                v = _np.ma.asarray(arr)
                if v.ndim == 2 and v.dtype.kind in "fiub" and v.size >= 4:
                    av = _np.ma.filled(v.astype(float), _np.nan)
                    if av.nbytes <= _MAX_BYTES:
                        import matplotlib as _mpl
                        _np.save(os.path.join(d, "fig_ax0_image0.npy"), av)
                        ent = {"name": "fig_ax0_image0", "kind": "array", "shape": list(av.shape),
                               "dtype": "float64", "source": "figure", "artist": "image", "via": "imsave",
                               "origin": kw.get("origin") or _mpl.rcParams["image.origin"]}
                        fin = av[_np.isfinite(av)]
                        lo = kw.get("vmin", float(fin.min()) if fin.size else None)
                        hi = kw.get("vmax", float(fin.max()) if fin.size else None)
                        if lo is not None and hi is not None:
                            ent["clim"] = [float(lo), float(hi)]
                        p = os.path.join(d, "index.json")
                        idx = json.load(open(p)) if os.path.exists(p) else []
                        idx.append(ent)
                        json.dump(idx, open(p, "w"), indent=0, default=str)
        except Exception:
            pass
        return _orig_imsave(fname, arr, *a, **kw)
    _mimg.imsave = _imsave
    _g["_cap_snapshot"] = _snap

    # Snapshot the locals (and return value) of user functions called from top level when
    # they return, so analyses done inside functions that only write files (no plot) are
    # still visible. Bounded to 30 snapshots.
    import sys as _sys

    def _prof(frame, event, arg):
        if event != "return" or frame.f_globals is not _g:
            return
        code = frame.f_code
        # skip module code, private helpers, lambdas/genexprs/comprehensions ('<...>') and
        # generator/coroutine frames (their 'return' events fire at every yield)
        if code.co_name.startswith(("<", "_")) or code.co_flags & (0x20 | 0x80 | 0x200):
            return
        b = frame.f_back
        if b is None or b.f_code.co_name != "<module>" or _state["r"] >= 30:
            return
        try:
            loc = dict(frame.f_locals)
            loc["returned"] = arg
            _state["r"] += 1
            _snap("ret%02d_%s" % (_state["r"], code.co_name[:40]), loc, with_globals=False)
        except Exception:
            pass
    _sys.setprofile(_prof)

_cap_install_capture()
del _cap_install_capture
'''


def drawn_patch_groups(snap_dir):
    """Hand-drawn patch groups of the figure saved in one snapshot directory (see _drawn_patches
    in PRELUDE): list of dicts with the record fields plus 'facecolors' (n x 4), 'bbox' (n x 4)
    and 'polygons' (list of n vertex arrays, or None when vertices were capped). [] if none."""
    import numpy as np
    p = os.path.join(snap_dir, "drawn_patches.json")
    if not os.path.exists(p):
        return []
    try:
        recs = json.load(open(p))
    except (OSError, ValueError):
        return []
    out = []
    for r in recs:
        try:
            g = dict(r)
            g["facecolors"] = np.load(os.path.join(snap_dir, r["facecolors"]))
            g["bbox"] = np.load(os.path.join(snap_dir, r["bbox"]))
            g["polygons"] = None
            if r.get("vertices") and r.get("vertices_complete"):
                vv = np.load(os.path.join(snap_dir, r["vertices"]))
                idx = vv[:, 0].astype(int)
                g["polygons"] = [vv[idx == q, 1:] for q in range(r["n"])]
            out.append(g)
        except Exception:
            continue
    return out


def run_with_capture(code, data_dir, work_dir, timeout=1200, memory=None, cpus="4", blocked=None,
                     cell_id="capture", blocks=None, image=None, skip=None):
    """Execute `code` (or, if given, the list `blocks` in order, tolerating failures exactly
    as the original run did) in a fresh container with capture; return a summary dict.
    `skip` = {block index: original outcome}: blocks the original run's container was killed in
    (OOM / crash / kill after timeout). They are not executed; their outcome is recorded as in the
    original. The original run's state restore after such a kill re-ran only the blocks that had
    ended normally, so the killed block left no state there either."""
    from container_sandbox import ContainerSandbox
    if memory is None:  # 8 GB normally, 16 GB for tasks with more than 300 MB of inputs
        size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(data_dir) for f in fs)
        memory = "16g" if size > 300 * 1024 * 1024 else "8g"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    extra = {"image": image} if image else {}
    sb = ContainerSandbox(work_dir, timeout, dataset_dir=data_dir, cell_id=cell_id, memory=memory,
                          cpus=cpus, blocked={} if blocked is None else blocked, exp="capture", **extra)
    try:
        pre = sb.execute(PRELUDE)
        if not pre.success:
            raise RuntimeError("capture prelude failed: " + pre.stderr[-500:])
        seed = sb.execute(SEED_PRELUDE)
        if not seed.success:
            raise RuntimeError("seed prelude failed: " + seed.stderr[-500:])
        # block_outcome per block: "ok" (ran to the end, success or exception), or
        # "timeout_interrupted" / "timeout_killed" / "oom" / "crash" (container_sandbox), so a
        # replay that was cut off somewhere else than the original can be flagged
        outs, block_success, block_outcome = [], [], []
        # full traceback head of each failed block (the agent's observation keeps only the tail of a
        # long stderr, which can lose the frame that locates the failing line)
        block_tb = []
        if blocks is None:
            r = sb.execute(code)
            n_ok = int(r.success)
            outs.append(r.stdout)
            block_success.append(bool(r.success))
            block_outcome.append(getattr(sb, "_last_outcome", "ok"))
        else:
            n_ok, r = 0, None
            for i, b in enumerate(blocks):
                if skip and i in skip:
                    outs.append("")
                    block_success.append(False)
                    block_outcome.append(skip[i])
                    block_tb.append(None)
                    continue
                r = sb.execute(b)
                n_ok += int(r.success)
                outs.append(r.stdout)
                block_success.append(bool(r.success))
                err = r.stderr or ""
                k = err.rfind("Traceback (most recent call last)")
                block_tb.append(err[k:k + 6000] if (not r.success and k >= 0) else None)
                block_outcome.append(getattr(sb, "_last_outcome", "ok"))
        sb.execute("_cap_snapshot('final')")
    finally:
        sb.close()
    with open(os.path.join(work_dir, "stdout.txt"), "w") as f:  # full stdout, all blocks in order
        f.write("\n".join(o for o in outs if o))
    snap_root = os.path.join(work_dir, "_task_root", "_snap")
    snaps = {}
    if os.path.isdir(snap_root):
        for tag in sorted(os.listdir(snap_root)):
            p = os.path.join(snap_root, tag, "index.json")
            if os.path.exists(p):
                snaps[tag] = json.load(open(p))
    return {"success": bool(r and r.success), "blocks_ok": n_ok,
            "stdout_tail": (r.stdout if r else "")[-3000:], "stderr_tail": (r.stderr if r else "")[-3000:],
            "outputs": sorted(os.listdir(os.path.join(work_dir, "pred_results"))),
            "snapshots": snaps, "events": sb.events, "block_success": block_success,
            "block_outcome": block_outcome, "block_traceback": block_tb, "step_timeout_s": timeout}
