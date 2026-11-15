"""T38 - Driving travel time from the Natural History Building (NHB), UIUC, to all nodes.

Gold: frozen OSM drive graph (240 nodes), osmnx speeds/travel times, origin = nearest node
to the NHB, Dijkstra on 'travel_time' (seconds); 239 nodes reachable.

Correctness (required):
  1. travel_time.png exists and is not blank;
  2. per-node travel times were computed: some captured per-node table / series (keyed by OSM
     node id, or values drawn in a scatter matched to node coordinates) agrees with the
     reference field (Pearson >= 0.90 on >= 80 % of nodes). The reference is the gold pipeline
     (osmnx speeds, Dijkstra on travel_time) run from the solution's own origin node (the node
     with the minimum value) when that origin passes check 4, else the gold-origin field
     (an accepted origin 229 m from the NHB gives r = 0.28 vs the gold-origin field, which
     would contradict check 4). Checks 3 and 7 use the same
     reference. A far-away origin (e.g. lat/lon swapped, 1.4 km off) fails check 4;
  3. the values are travel *times*: median ratio to gold within 0.67-1.5 in seconds or in
     minutes. Network *length* (metres) correlates 0.99 with time but fails here;
  4. the origin (node with the minimum value, ~0) lies within 400 m of the NHB;
  5. network routing was done: from the solution's own origin node, the values correlate with
     the network travel-time field (Dijkstra on osmnx travel times) at Pearson >= 0.97 AND
     better than with the straight-line distance from that origin (by >= 0.01). Straight-line
     distance / speed reaches 0.954 vs gold but 1.0 vs straight line; legitimate variants
     (custom speeds 0.994, undirected 0.997, centre origin 1.0 vs its own field) pass;
  6. speeds were derived from the graph: the values are not a pure network-length field
     (Pearson with length-Dijkstra from the own origin, directed or undirected, < 0.999).
     A single assumed speed for every edge gives exactly 1.000; osmnx / posted-speed variants
     0.989-0.993;
  7. the figure colours the nodes by travel time: a drawn point collection whose offsets are the
     graph nodes carries values (or face colours inverted to colormap positions) with
     |Spearman| >= 0.90 against the reference field of check 2. Node positions are tried in
     lon/lat (1e-4 deg) and projected to the local UTM zone, Web Mercator and
     the CRSs of the run (1 m).
"""
import numpy as np

from checklib import file_exists, png_not_blank, result

NHB = (40.109442588847976, -88.22625178710955)  # lat, lon


def _graph_nodes(ctx):
    import osmnx as ox
    G = ox.load_graphml(f"{ctx.inputs_dir}/uiuc_drive_network.graphml")
    ids = np.array(list(G.nodes), dtype="int64")
    xy = np.array([[G.nodes[n]["x"], G.nodes[n]["y"]] for n in G.nodes])
    return ids, xy


_G = {}


def _own_fields(ctx, origin):
    """Fields from the solution's own origin: network travel time, network length (directed and
    undirected) and straight-line distance."""
    import networkx as nx
    import osmnx as ox
    if ctx.inputs_dir not in _G:
        g = ox.load_graphml(f"{ctx.inputs_dir}/uiuc_drive_network.graphml")
        _G[ctx.inputs_dir] = ox.add_edge_travel_times(ox.add_edge_speeds(g))
    G = _G[ctx.inputs_dir]
    T = nx.single_source_dijkstra_path_length(G, origin, weight="travel_time")
    L1 = nx.single_source_dijkstra_path_length(G, origin, weight="length")
    L2 = nx.single_source_dijkstra_path_length(G.to_undirected(), origin, weight="length")
    y0, x0 = G.nodes[origin]["y"], G.nodes[origin]["x"]
    S = {n: float(np.hypot((G.nodes[n]["y"] - y0) * 111000, (G.nodes[n]["x"] - x0) * 111000 * np.cos(np.radians(y0))))
         for n in G.nodes}
    return T, (L1, L2), S


def _node_position_variants(ctx, xy):
    """Node positions as lon/lat (tolerance 1e-4 deg) and reprojected to the local UTM
    zone, Web Mercator and every CRS seen in the run (tolerance 1 m): a map drawn after projecting
    the graph (e.g. to UTM 16N) does not match the lon/lat positions."""
    from pyproj import CRS, Transformer
    out = [("lon/lat", xy, 1e-4)]
    lon0, lat0 = float(np.mean(xy[:, 0])), float(np.mean(xy[:, 1]))
    utm = (32600 if lat0 >= 0 else 32700) + int((lon0 + 180) // 6) + 1
    crss = {f"EPSG:{utm}", "EPSG:3857"}
    for _, m, _ in ctx.objects("pred", include_files=False):
        c = m.get("crs")
        if c and c != "None":
            crss.add(c)
    for c in sorted(crss):
        try:
            cc = CRS.from_user_input(c)
            if not cc.is_projected:
                continue
            px, py = Transformer.from_crs("EPSG:4326", cc, always_xy=True).transform(xy[:, 0], xy[:, 1])
            P = np.c_[px, py]
            if np.isfinite(P).all():
                out.append((c, P, 1.0))
        except Exception:
            continue
    return out


def _drawn_node_colours(ctx, ids, xy, gold):
    """Best |Spearman| between gold times and the values / colour positions of drawn points at node positions."""
    from scipy.spatial import cKDTree
    from checklib import drawn_arrays, _cmap_positions
    variants = [(lab, cKDTree(P), tol) for lab, P, tol in _node_position_variants(ctx, xy)]
    arrs = {}
    for tag, meta, a in drawn_arrays(ctx):
        n = meta["name"]
        for suf in ("_values", "_facecolors", "_offsets"):
            if n.endswith(suf):
                arrs.setdefault((tag, n[: -len(suf)]), {})[suf] = np.asarray(a, float)
    best = (None, -1.0)
    for (tag, base), d in arrs.items():
        if "_offsets" not in d:
            continue
        o_all = d["_offsets"].reshape(-1, 2)
        fin = np.isfinite(o_all).all(1)     # NaN / inf offsets (e.g. masked points) crash cKDTree.query
        o = o_all[fin]
        if len(o) == 0:
            continue
        hit = None
        for vlab, tree, tol in variants:
            dist, idx = tree.query(o)
            ok = dist < tol
            if ok.sum() >= 0.5 * len(gold):
                hit = vlab
                break
        if hit is None:
            continue
        cands = []
        if "_values" in d and len(d["_values"].ravel()) == len(o_all):
            cands.append(("values", d["_values"].ravel()[fin]))
        if "_facecolors" in d and len(d["_facecolors"]) == len(o_all) and \
                len(np.unique(np.round(d["_facecolors"][fin][:, :3], 4), axis=0)) >= 3:
            pos, cname, res = _cmap_positions(d["_facecolors"][fin])
            if pos is not None and res <= 0.03:
                cands.append((f"colours ({cname})", pos))
        for kind, v in cands:
            vec = {int(ids[i]): float(x) for i, x, k in zip(idx, v, ok) if k and np.isfinite(x)}
            cm = [k for k in gold if k in vec]
            if len(cm) < 0.8 * len(gold):
                continue
            a = np.array([vec[k] for k in cm]); g = np.array([gold[k] for k in cm])
            if a.std() == 0:
                continue
            from scipy.stats import spearmanr
            rho = abs(float(spearmanr(a, g).correlation))
            if rho > best[1]:
                best = (f"{base}@{tag} {kind}" + ("" if hit == "lon/lat" else f" (node positions in {hit})"), rho)
    return best


def _r(vec, ref):
    cm = [k for k in ref if k in vec and np.isfinite(vec[k]) and np.isfinite(ref[k])]
    if len(cm) < 20:
        return float("nan")
    a = np.array([vec[k] for k in cm], float); b = np.array([ref[k] for k in cm], float)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")


def _gold_times(ctx):
    for tag, meta, load in ctx.objects("gold"):
        if meta["name"] == "nodes" and meta.get("kind") in ("gdf", "df"):
            d = load()
            d = d[d["time2nhb"].notna()]
            return dict(zip(d["osmid"].astype("int64"), d["time2nhb"].astype(float)))
    return None


def _pred_vectors(ctx, ids, xy):
    """Yield (label, {node_id: value}) from captured tables, series and scatter data."""
    idset = set(ids.tolist())
    n = len(ids)
    for tag, meta, load in ctx.objects("pred"):
        k = meta.get("kind")
        if k in ("gdf", "df", "series") and 0.5 * n <= meta.get("rows", 0) <= 1.1 * n:
            try:
                d = load()
            except Exception:
                continue
            keycol = None
            for c in ("osmid", "node", "node_id", "nodes", "index", "id", "level_0"):
                if c in d.columns:
                    import pandas as pd
                    vals = pd.to_numeric(d[c], errors="coerce")  # graph/dict captures store ids as str
                    if vals.notna().mean() > 0.8 and vals.dropna().astype("int64").isin(idset).mean() > 0.8:
                        d = d.assign(**{c: vals})
                        keycol = c
                        break
            if keycol is None:
                continue
            for c in d.columns:
                if c in (keycol, "geometry", "x", "y", "lon", "lat", "street_count") or d[c].dtype.kind not in "fi":
                    continue
                m = d[keycol].notna()
                yield f"{meta['name']}.{c}@{tag}", dict(zip(d.loc[m, keycol].astype("int64"), d.loc[m, c].astype(float)))
    # scatter: values + offsets drawn in the figure, matched to node coordinates
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    objs = {(tag, meta["name"]): (meta, load) for tag, meta, load in ctx.objects("pred", "array")}
    for (tag, name), (meta, load) in objs.items():
        if not name.endswith("_values"):
            continue
        off = objs.get((tag, name[:-7] + "_offsets"))
        if off is None:
            continue
        try:
            v = np.asarray(load(), float).ravel()
            o = np.asarray(off[1](), float)
        except Exception:
            continue
        if len(v) != len(o) or len(v) < 0.5 * n:
            continue
        o = o.reshape(len(v), -1)[:, :2]
        fin = np.isfinite(o).all(1)         # NaN / inf offsets crash cKDTree.query
        if fin.sum() < 0.5 * n:
            continue
        v, o = v[fin], o[fin]
        dist, idx = tree.query(o)
        ok = dist < 1e-4
        if ok.mean() < 0.8:
            continue
        yield f"{name}@{tag}", dict(zip(ids[idx[ok]], v[ok]))


def checks(ctx):
    out = []
    names = ["travel_time.png", "Travel_Time.png", "travel_time.PNG"]
    r = file_exists(ctx, names)
    out.append(r)
    if r["passed"]:
        nb, std = png_not_blank(ctx.pred(r["value"][0]))
        out.append(result("figure not blank", nb, value=std, threshold=">=3 grey-level std"))
    else:
        out.append(result("figure not blank", False, note="no travel_time.png"))

    try:
        ids, xy = _graph_nodes(ctx)
        gold = _gold_times(ctx)
    except Exception as e:
        ids, gold = None, None
    names3 = ["per-node travel times match gold (Pearson >= 0.90)", "values are travel times (median ratio 0.67-1.5, s or min)",
              "origin within 400 m of Natural History Building",
              "network routing (r with own-origin network time >= 0.97 and > straight-line r + 0.01)",
              "speeds derived from the graph (not a pure network-length field, r < 0.999)",
              "figure colours nodes by travel time (|Spearman| >= 0.90 vs gold)"]
    info = None
    if gold is None:
        for nm in names3:
            out.append(result(nm, False, note="gold node table missing"))
        return out
    # Reference field: the task fixes the origin only as "the Natural History Building"; an
    # origin within 400 m (check 4) is accepted, and the time field is then the gold pipeline
    # (osmnx speeds, Dijkstra on travel_time) run from the solution's OWN origin node. A
    # comparison with the gold-origin field would on this 1 km graph in effect require the gold
    # origin node (an accepted origin 229 m away gives r = 0.28).
    pos_of = dict(zip(ids.tolist(), xy.tolist()))
    own = {}

    def ref_for(vec):
        o = min((k for k in vec if np.isfinite(vec[k])), key=lambda k: vec[k])
        pos = pos_of.get(int(o))
        dm = None if pos is None else float(np.hypot((pos[1] - NHB[0]) * 111000,
                                                     (pos[0] - NHB[1]) * 111000 * np.cos(np.radians(NHB[0]))))
        if dm is not None and dm <= 400:
            if o not in own:
                try:
                    own[o] = _own_fields(ctx, int(o))
                except Exception:
                    own[o] = None
            if own[o] is not None:
                return o, dm, own[o][0], "own origin"
        return o, dm, gold, "gold origin"

    best = (None, -2.0, None, None)
    for lab, vec in _pred_vectors(ctx, ids, xy):
        try:
            o, dm, ref, how = ref_for(vec)
        except ValueError:
            continue
        common = [k for k in ref if k in vec and np.isfinite(vec[k])]
        if len(common) < 0.8 * len(gold):
            continue
        a = np.array([vec[k] for k in common]); g = np.array([ref[k] for k in common], float)
        if a.std() == 0:
            continue
        rr = float(np.corrcoef(a, g)[0, 1])
        if rr > best[1]:
            best = (lab, rr, vec, (o, dm, ref, how))
    lab, rr, vec, info = best
    out.append(result(names3[0], rr >= 0.90, value=round(rr, 3) if lab else None, threshold=0.90,
                      note=(f"{lab}; vs {info[3]} field" if lab else
                            "no per-node numeric result keyed by node id / drawn at node positions")))
    if lab and rr >= 0.90:
        o, dm, ref, how = info
        common = [k for k in ref if k in vec and ref[k] > 0 and np.isfinite(vec[k])]
        ratio = float(np.median([vec[k] / ref[k] for k in common]))
        ok = 0.67 <= ratio <= 1.5 or 0.67 <= ratio * 60 <= 1.5
        out.append(result(names3[1], ok, value=round(ratio, 3), threshold="0.67-1.5 (or x60 for minutes)"))
        if dm is None:
            out.append(result(names3[2], False, note="origin node unknown"))
        else:
            out.append(result(names3[2], dm <= 400, value=round(dm, 1), threshold=400, note=f"origin node {o}"))
        try:
            T, Ls, S = _own_fields(ctx, int(o))
            rt, rs = _r(vec, T), _r(vec, S)
            rl = max(_r(vec, L) for L in Ls)
            out.append(result(names3[3], rt >= 0.97 and rt > rs + 0.01, value=f"r_network={rt:.4f}, r_straight={rs:.4f}",
                              threshold=">= 0.97 and > r_straight + 0.01", note=lab))
            out.append(result(names3[4], rl < 0.999, value=round(rl, 4), threshold="< 0.999", note=lab))
        except Exception as e:
            out.append(result(names3[3], False, note=f"own-origin fields failed: {e!r}"[:200]))
            out.append(result(names3[4], False, note=f"own-origin fields failed: {e!r}"[:200]))
    else:
        for n in names3[1:5]:
            out.append(result(n, False, note="no matching travel-time vector"))
    try:
        fref = info[2] if (lab and rr >= 0.90) else gold   # same reference field as check 2
        dl, drho = _drawn_node_colours(ctx, ids, xy, fref)
    except Exception as e:
        dl, drho = None, -1.0
    if dl is None:
        out.append(result(names3[5], False, note="UNVERIFIABLE: no drawn point collection at node positions with values/colours"))
        out[-1]["unverifiable"] = True
    else:
        out.append(result(names3[5], drho >= 0.90, value=round(drho, 3), threshold=0.90, note=dl))
    return out
