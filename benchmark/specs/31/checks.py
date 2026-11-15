"""T31 - Flow L-function for OD flows (aggregation scale of geographical flows).

Flow distance = max(origin distance, destination distance); under CSR K(r) = pi^2 r^4 and
L(r) = (K/pi^2)^(1/4) - r = 0. Gold: translation edge correction, i != j, global K = mean of
local K, 99 CSR simulations for the envelope. The supplied flows are themselves CSR (uniform
origins and destinations in the unit square): L(r) stays within +/-0.007 of 0 and inside the
envelope at every scale - there is no significant aggregation.

Correctness (required):
  1. a non-blank figure;
  2. the plotted L(r) curve (figure line or scatter offsets with >= 10 points; only drawn
     curves count, because captured function returns include simulated-CSR curves)
     agreeing with the gold L over r in [0.05, 0.35]: max |dL| <= 0.012 and mean |dL| <= 0.005.
     No edge correction (L drifts to -0.02 at r=0.2, -0.06 at r=0.35), counting each flow as
     its own neighbour (+0.04 at r=0.06), a 2-D L transform (sqrt(K/pi)) or plotting K fail.
     The curve must reach r >= 0.2; compared on [0.05, min(0.35, r_max)];
  2b. the K behind it is the FLOW K: a captured K vector on an r grid (or the column mean of a
     per-flow local-K matrix) within 10 % mean relative error of the gold K (~pi^2 r^4) -
     a planar Ripley K of points (~pi r^2) fails. A CSR-normalised (observed / edge-corrected
     expected counts) flow K is also accepted when the per-flow 4-D expectation array is
     captured (_ratio_form_k); a planar Ripley K normalised to ~1 fails. Also accepted when a
     captured per-flow local-K matrix is verified against max-distance neighbour counts the
     grader recomputes from flow.csv (_local_k_data_route) and its mean, read as K, K/(pi r^2)
     or K/K_CSR, matches the gold K (same 10 %).
  4. confidence intervals (required: the text
     asks to "visualize the L-function with confidence intervals"): code that simulates random
     flows for an envelope (fake noise bands `randn(len(L...` do not count), OR a drawn band
     in the L figure - a fill_between polygon or two further non-constant line artists - that
     brackets the gold L or the drawn L (a CI around the estimate) at >= 80 % of r in [0.05, min(0.35, r_max)] with both bounds within
     +/-0.03 (gold 99-simulation envelope: +/-0.002-0.006; a 0.05*randn noise band fails).
Reported, not required:
  3. local K computed per flow (an n x m array, n = 1000) - the text describes the route, but a
     global K summed over pairs is the same result (correct_ripley_isotropic has no per-flow matrix);
  5. conclusion consistent with the data: no claimed significant clustering scale (not coded).
"""
import os

import numpy as np

from checklib import result, png_not_blank

R_LO, R_HI = 0.05, 0.35


def _gold_curve(ctx):
    import pandas as pd
    g = pd.read_csv(ctx.gold("flow_L_function.csv"))
    return g["r"].to_numpy(), g["L"].to_numpy()


def _candidates(ctx, with_matrix_means=False):
    arrays = []
    for tag, meta, load in ctx.objects("pred", "array"):
        shp = meta.get("shape", [])
        try:
            a = np.asarray(load(), float)
        except Exception:
            continue
        arrays.append((meta["name"] + "@" + tag, a))
    out = []
    vecs = [(n, a) for n, a in arrays if a.ndim == 1 and len(a) >= 10]
    if with_matrix_means:   # per-flow local K (n x m): its column mean is the global K
        vecs += [(n + ".mean0", np.nanmean(a, 0)) for n, a in arrays
                 if a.ndim == 2 and a.shape[0] >= 900 and 10 <= a.shape[1] < a.shape[0]]
    for n, a in arrays:
        if a.ndim == 2 and a.shape[1] == 2 and a.shape[0] >= 10:
            out.append((n, a[:, 0], a[:, 1]))
        if a.ndim == 2 and a.shape[0] == 2 and a.shape[1] >= 10:
            out.append((n, a[0], a[1]))
    def is_grid(a):  # a distance grid: increasing and evenly (or geometrically) spaced
        d = np.diff(a)
        if not (np.all(d > 0) and 0 <= a[0] < 0.2 and a[-1] <= 3):
            return False
        q = d[1:] / d[:-1]
        return bool(np.allclose(d, d.mean(), rtol=0.02) or np.allclose(q, q.mean(), rtol=0.02))
    xs = [(n, a) for n, a in vecs if is_grid(a)]
    for xn, x in xs:
        for yn, y in vecs:
            if yn != xn and len(y) == len(x):
                out.append((f"{yn} vs {xn}", x, y))
    return out


def _compare(x, y, gr, gl):
    # r_max >= 0.2: r_max = 1/4 of the side is the textbook rule
    if not np.all(np.isfinite(x)) or x.min() > R_LO or x.max() < 0.2 or np.any(np.diff(np.sort(x)) <= 0):
        return None
    o = np.argsort(x)
    x, y = x[o], y[o]
    m = (gr >= R_LO) & (gr <= min(R_HI, x.max()))
    p = np.interp(gr[m], x, y)
    if not np.all(np.isfinite(p)) or np.std(y) == 0:
        return None
    d = np.abs(p - gl[m])
    return float(d.max()), float(d.mean())


def _ratio_form_k(ctx, gr, gk):
    """The flow K in CSR-normalised (ratio) form.

    A legitimate flow K can be stored as observed / CSR-expected neighbour counts per flow
    (expected = (n-1)|disc(o,r) & O|/|O| * |disc(d,r) & D|/|D|, an edge correction), whose
    flow mean is ~1 under CSR, or rescaled by pi r^2 for a planar-style L. Accepted only with
    evidence that the normalisation is the FLOW one: a captured per-flow expectation array
    (n x m or m x n, m = r grid length) on the 4-D scale (flow mean / (c pi^2 r^4) in
    [0.5, 1.05] for r in [0.05, 0.15], c in {1, n-1, n}), an integer count array of the same
    shape, and a K vector proportional (CV <= 2 %) to the flow mean of counts / expectation.
    The vector, read as K/K_CSR (or K/(pi r^2)), times pi^2 r^4 must then match the gold K
    (mean rel. error <= 10 %). A planar Ripley K normalised to ~1 has no 4-D expectation
    array and fails (control wrong_planar_ripley_normalised).
    Returns (note, rel_err) or None."""
    arrs = []
    for tag, meta, load in ctx.objects("pred", "array"):
        try:
            a = np.asarray(load(), float)
        except Exception:
            continue
        if a.ndim in (1, 2):
            arrs.append((meta["name"] + "@" + tag, a))
    n = 1000
    best = None
    def is_grid(a):
        d = np.diff(a)
        return bool(a.ndim == 1 and len(a) >= 10 and np.all(np.isfinite(a)) and np.all(d > 0)
                    and 0 <= a[0] <= R_LO and 0.2 <= a[-1] <= 3)
    for xn, x in arrs:
        if not is_grid(x):
            continue
        m = len(x)
        def orient(a):
            if a.ndim == 2 and a.shape == (n, m):
                return a
            if a.ndim == 2 and a.shape == (m, n):
                return a.T
            return None
        mats = [(nm, orient(a)) for nm, a in arrs if orient(a) is not None]
        sel = (x >= R_LO) & (x <= 0.15)
        if sel.sum() < 2:
            continue
        exps = []
        for en, E in mats:
            with np.errstate(all="ignore"):
                eb = np.nanmean(np.where(E > 0, E, np.nan), 0)
                for c in (1.0, n - 1.0, float(n)):
                    q = eb[sel] / (c * np.pi ** 2 * x[sel] ** 4)
                    if np.all(np.isfinite(q)) and 0.5 <= np.median(q) <= 1.05 and q.min() >= 0.4 and q.max() <= 1.1:
                        exps.append((en, E))
                        break
        cnts = [(cn, C) for cn, C in mats
                if np.all(np.isfinite(C)) and C.min() >= 0 and np.allclose(C, np.round(C)) and C.max() >= 1]
        msk = (gr >= R_LO) & (gr <= min(R_HI, x.max()))
        cmp = (x >= R_LO) & (x <= min(R_HI, x.max()))
        for en, E in exps:
            for cn, C in cnts:
                if C is E:
                    continue
                with np.errstate(all="ignore"):
                    ratio = np.nanmean(np.where(E > 0, C / E, np.nan), 0)
                for yn, y in arrs:
                    if y.ndim != 1 or len(y) != m or yn == xn:
                        continue
                    for conv, lab in ((1.0, "K/K_CSR"), (1 / (np.pi * x ** 2), "K/(pi r^2)")):
                        yv = y * conv
                        ok = cmp & np.isfinite(yv) & np.isfinite(ratio) & (ratio > 0)
                        if ok.sum() < max(5, 0.8 * cmp.sum()):
                            continue
                        q = yv[ok] / ratio[ok]
                        if not (np.all(q > 0) and np.std(q) / np.mean(q) <= 0.02):
                            continue
                        kk = yv * np.pi ** 2 * x ** 4
                        f = np.isfinite(kk)
                        pk = np.interp(gr[msk], x[f], kk[f])
                        rel = float(np.mean(np.abs(pk / gk[msk] - 1)))
                        if best is None or rel < best[1]:
                            best = (f"{yn} as {lab} = mean({cn}/{en}) x pi^2 r^4 vs {xn} (ratio-form flow K)", rel)
    return best


def _local_k_data_route(ctx, gr, gk):
    """A flow K normalised analytically - local K per
    flow K_i(r) = cnt_i(r) * w_i(r) / ((n-1) pi r^2) with edge weights w = 1/(p_o p_d) - stores
    K/(pi r^2) and leaves no expectation array, so _ratio_form_k cannot see it. Data-level
    evidence instead: a captured per-flow matrix (n x m or m x n on a captured r grid, n = 1000)
    is validated against the max-distance neighbour counts (i != j) the grader recomputes from
    dataset/flow.csv: (i) zero pattern identical (K_i(r) > 0 exactly where the count is > 0) at
    >= 99.5 % of the cells for r in [0.05, 0.35] (the right method gives 1.000; 4-D Euclidean
    0.943, origin-only or self-pairs 0.836); (ii) Spearman >= 0.8 with the counts at r ~ 0.1;
    (iii) per r, the ratio K_i / count_i is a bounded edge weight (max/min <= 20). The column
    mean, read as K, K/(pi r^2) or K/K_CSR, times the factor must match the gold K (mean rel.
    error <= 10 %). Returns (note, rel_err) or None."""
    import pandas as pd
    from scipy.stats import spearmanr
    try:
        f = pd.read_csv(os.path.join(ctx.inputs_dir, "flow.csv"))[["x_o", "y_o", "x_d", "y_d"]].to_numpy(float)
    except Exception:
        return None
    n = len(f)
    D = np.maximum(np.hypot(f[:, None, 0] - f[None, :, 0], f[:, None, 1] - f[None, :, 1]),
                   np.hypot(f[:, None, 2] - f[None, :, 2], f[:, None, 3] - f[None, :, 3]))
    np.fill_diagonal(D, np.inf)
    Ds = np.sort(D, axis=1)
    arrs = []
    for tag, meta, load in ctx.objects("pred", "array", include_files=False):
        sh = meta.get("shape") or []
        if meta.get("source") == "figure" or len(sh) not in (1, 2) or (len(sh) == 2 and n not in sh):
            continue
        try:
            a = np.asarray(load(), float)
        except Exception:
            continue
        arrs.append((meta["name"] + "@" + tag, a))
    grids = [(xn, x) for xn, x in arrs if x.ndim == 1 and len(x) >= 10 and np.all(np.isfinite(x))
             and np.all(np.diff(x) > 0) and 0 <= x[0] <= R_LO and 0.2 <= x[-1] <= 3]
    best, seen = None, set()
    for xn, x in grids:
        m = len(x)
        mats = [(nm, a if a.shape == (n, m) else a.T) for nm, a in arrs
                if a.ndim == 2 and a.shape in ((n, m), (m, n))]
        if not mats:
            continue
        sel = (x >= R_LO) & (x <= R_HI)
        if sel.sum() < 3:
            continue
        C = np.stack([np.searchsorted(Ds[i], x, side="right") for i in range(n)])   # n x m counts
        j1 = int(np.argmin(np.abs(x - 0.1)))
        msk = (gr >= R_LO) & (gr <= min(R_HI, x.max()))
        for mn, M in mats:
            key = (xn.split("@")[0], mn.split("@")[0], float(np.nansum(M)))
            if key in seen:
                continue
            seen.add(key)
            Ms, Cs = M[:, sel], C[:, sel]
            if not np.all(np.isfinite(Ms)) or Ms.min() < 0:
                continue
            zero = float(np.mean((Ms > 0) == (Cs > 0)))
            if zero < 0.995 or np.std(M[:, j1]) == 0:
                continue
            rho = float(spearmanr(M[:, j1], C[:, j1]).correlation)
            if not rho >= 0.8:
                continue
            with np.errstate(all="ignore"):
                q = np.where(Cs > 0, Ms / Cs, np.nan)
                spread = np.nanmax(q, 0) / np.nanmin(q, 0)
            if np.any(np.nanmin(q, 0) <= 0) or np.nanmax(spread) > 20:
                continue
            km = M.mean(0)
            for conv, lab in ((np.ones(m), "K"), (np.pi * x ** 2, "K/(pi r^2)"), (np.pi ** 2 * x ** 4, "K/K_CSR")):
                kk = km * conv
                pk = np.interp(gr[msk], x, kk)
                rel = float(np.mean(np.abs(pk / gk[msk] - 1)))
                if best is None or rel < best[1]:
                    best = (f"mean of per-flow {mn} as {lab} vs {xn} (per-flow counts verified: zero pattern "
                            f"{zero:.3f}, rho {rho:.2f} at r~0.1)", rel)
    return best


def checks(ctx):
    out = []
    pngs = ctx.pred_files("*.png")
    out.append(result("figure written (non-blank PNG)", any(png_not_blank(ctx.pred(p))[0] for p in pngs), value=pngs))
    try:
        gr, gl = _gold_curve(ctx)
    except Exception as e:
        out.append(result("L(r) curve matches gold", False, note=f"gold missing: {e!r}"[:200]))
        return out
    # The L curve is taken from what the figure DRAWS (line artists, scatter offsets):
    # function-return captures also hold L curves of simulated CSR
    # flows (envelope runs), which are ~0 like the gold and must not stand in for the
    # observed curve; the task's deliverable is the plotted L(r).
    best = None
    cands = [(n, x, y) for n, x, y in _candidates(ctx) if n.startswith("fig_ax")]
    for name, x, y in cands:
        c = _compare(x, y, gr, gl)
        if c and (best is None or c[0] < best[1][0]):
            best = (name, c)
    if best is None:
        out.append(result("L(r) curve matches gold", False,
                          note=f"no drawn (r, L) curve covering r in [{R_LO}, 0.2]"))
    else:
        name, (mx, mn) = best
        out.append(result("L(r) curve matches gold", mx <= 0.012 and mn <= 0.005,
                          value={"max_abs": round(mx, 4), "mean_abs": round(mn, 4)},
                          threshold="max<=0.012, mean<=0.005 on r in [0.05, 0.35]", note=name))

    # flow K-function: on these CSR data any unbiased L of any CSR
    # pattern is ~0 (e.g. a planar Ripley L of the origins), so the L curve alone cannot show
    # that the FLOW K was computed. Require a captured K curve (vector on an r grid, or the
    # column mean of a per-flow local-K matrix) that matches the gold flow K (~pi^2 r^4) -
    # mean relative error <= 10 % on r in [0.05, min(0.35, r_max)]. A planar K (~pi r^2) is
    # off by orders of magnitude.
    import pandas as pd
    gk = pd.read_csv(ctx.gold("flow_L_function.csv"))["K"].to_numpy()
    kbest = None
    for name, x, y in _candidates(ctx, with_matrix_means=True):
        try:
            if not np.all(np.isfinite(x)) or x.min() > R_LO or x.max() < 0.2 or not np.all(np.isfinite(y)):
                continue
            o = np.argsort(x)
            m = (gr >= R_LO) & (gr <= min(R_HI, x.max()))
            pk = np.interp(gr[m], x[o], y[o])
            # the task text defines the flow distance (both origins AND both destinations
            # within r = max distance). The 4-D Euclidean flow distance (K_CSR = pi^2 r^4 / 2)
            # is a different method and is not accepted
            # (control wrong_method_euclidean4d_flow_distance).
            for fac, lab in ((1.0, "max-distance flow K"),):
                rel = float(np.mean(np.abs(pk / (fac * gk[m]) - 1)))
                if kbest is None or rel < kbest[1]:
                    kbest = (f"{name} ({lab})", rel)
        except Exception:
            continue
    rk = _ratio_form_k(ctx, gr, gk)
    if rk and (kbest is None or kbest[1] > 0.10) and rk[1] <= 0.10:
        kbest = rk
    if kbest is None or kbest[1] > 0.10:
        lr = _local_k_data_route(ctx, gr, gk)
        if lr and lr[1] <= 0.10:
            kbest = lr
    out.append(result("flow K-function (4-D, ~pi^2 r^4) computed - matches gold K", bool(kbest and kbest[1] <= 0.10),
                      value=round(kbest[1], 4) if kbest else None, threshold="mean rel. error <= 0.10 vs gold K (max-distance flow distance, as defined in the task text)",
                      note=kbest[0] if kbest else "no K curve or local-K matrix on an r grid captured"))

    lk = any(len(meta.get("shape", [])) == 2 and 1000 in meta["shape"] and min(meta["shape"]) >= 10
             and meta["shape"] != [1000, 4] for tag, meta, load in ctx.objects("pred", "array"))
    out.append(result("local K computed per flow (1000 x m array)", lk, required=False))

    code = ctx.code_text()
    sim = any(k in code for k in ("uniform(", "random.rand", "random(", "rng.")) and \
        any(k in code for k in ("simul", "envelope", "percentile", "quantile", "n_sim", "nsim", "permutation",
                                "monte", "Monte"))
    fake = "randn(len(L" in code or "randn(len(l" in code.lower()
    obs = None
    if best is not None:
        for n_, x_, y_ in cands:
            if n_ == best[0]:
                o_ = np.argsort(x_)
                obs = (x_[o_], y_[o_])
    band = _drawn_band(ctx, gr, gl, best[0] if best else None, obs)
    out.append(result("confidence envelope (CSR simulation or drawn band around L)", (sim and not fake) or bool(band),
                      value={"simulation_code": sim and not fake, "drawn_band": band or None},
                      note="required (the text asks for confidence intervals)"))
    return out


def _drawn_band(ctx, gr, gl, l_name, obs=None):
    """A band drawn in the figure around the L curve: a fill_between polygon (vertices binned on
    the gold r grid: max / min y) or a pair of further non-constant line artists. Returns a
    description or None. Bounds must bracket the gold L or the drawn observed L (obs: a CI
    around the estimate, e.g. delta-method SE) at >= 80 % of r in [0.05, min(0.35, r_max)] and
    stay within +/-0.03 (a 0.05*randn noise band does not)."""
    def ok(xg, lo, hi):
        m = np.isfinite(lo) & np.isfinite(hi)
        if m.sum() < 5:
            return False
        within = (np.abs(lo[m]) <= 0.03) & (np.abs(hi[m]) <= 0.03)
        if within.mean() < 0.8:
            return False
        refs = [np.interp(xg, gr, gl)]
        if obs is not None and obs[0].max() >= xg.max() - 1e-9:
            refs.append(np.interp(xg, obs[0], obs[1]))
        for g in refs:
            br = (lo[m] <= g[m] + 1e-9) & (hi[m] >= g[m] - 1e-9) & (hi[m] - lo[m] > 1e-5)
            if br.mean() >= 0.8:
                return True
        return False
    lines, polys = [], []
    for tag, meta, load in ctx.objects("pred", "array", include_files=False):
        if meta.get("source") != "figure":
            continue
        nm = meta["name"]
        try:
            if "_line" in nm and meta.get("artist") == "line":
                a = np.asarray(load(), float)
                if a.ndim == 2 and a.shape[1] == 2 and len(a) >= 5 and f"{nm}@{tag}" != l_name:
                    lines.append((tag, nm, a))
            elif nm.endswith("_vertices") and "Poly" in str(meta.get("artist", "")):
                polys.append((tag, nm, np.asarray(load(), float)))
        except Exception:
            continue
    for tag, nm, v in polys:
        if v.ndim != 2 or v.shape[1] != 3:
            continue
        x, y = v[:, 1], v[:, 2]
        f = np.isfinite(x) & np.isfinite(y)
        x, y = x[f], y[f]
        if not len(x) or x.max() < 0.2:
            continue
        xg = gr[(gr >= R_LO) & (gr <= min(R_HI, x.max()))]
        h = max(np.median(np.diff(np.unique(x))) if len(np.unique(x)) > 1 else 0.005, 1e-4) * 0.6
        lo = np.array([y[np.abs(x - q) <= h].min() if (np.abs(x - q) <= h).sum() >= 2 else np.nan for q in xg])
        hi = np.array([y[np.abs(x - q) <= h].max() if (np.abs(x - q) <= h).sum() >= 2 else np.nan for q in xg])
        if ok(xg, lo, hi):
            return f"fill polygon {nm}@{tag}"
    ls = [(t, n, a) for t, n, a in lines if np.nanstd(a[:, 1]) > 0 and np.all(np.diff(a[:, 0]) > 0)]
    for i in range(len(ls)):
        for j in range(i + 1, len(ls)):
            (t1, n1, a1), (t2, n2, a2) = ls[i], ls[j]
            if t1 != t2:
                continue
            xm = min(a1[:, 0].max(), a2[:, 0].max())
            if xm < 0.2:
                continue
            xg = gr[(gr >= R_LO) & (gr <= min(R_HI, xm))]
            y1, y2 = np.interp(xg, a1[:, 0], a1[:, 1]), np.interp(xg, a2[:, 0], a2[:, 1])
            if ok(xg, np.minimum(y1, y2), np.maximum(y1, y2)):
                return f"line pair {n1}, {n2}@{t1}"
    return None
