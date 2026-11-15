"""Correct: the flow K is computed as pi r^2 * N / (n |disc&O| |disc&D|), a CSR-normalised flow K on the
planar scale. The script is a sequence of blocks run in
order; blocks that raise an error are wrapped in try/except.
"""


import os; os.makedirs('pred_results', exist_ok=True)

# --- block 1 ---

import pandas as pd, numpy as np

flow = pd.read_csv('dataset/flow.csv')
o_area = pd.read_csv('dataset/o_area.csv')
d_area = pd.read_csv('dataset/d_area.csv')

print("FLOW:", flow.shape, flow.dtypes.to_dict())
print(flow.head())
print(flow.describe().T[['min','max','mean']])
print("missing:", flow.isna().sum().sum())
print()
print("O_AREA:\n", o_area)
print()
print("D_AREA:\n", d_area)

# --- block 2 (wrapped in try/except) ---
try:

    import numpy as np, pandas as pd
    from scipy.spatial.distance import cdist

    o = flow[['x_o','y_o']].values
    d = flow[['x_d','y_d']].values

    # uniformity check: mean & spread, quadrant counts
    print("origin mean/nd:  ", o.mean(axis=0).round(3), o.std(axis=0).round(3))
    print("dest  mean/nd:  ", d.mean(axis=0).round(3), d.std(axis=0).round(3))
    for name, pts in [("origins", o), ("dests", d)]:
        q = pd.cut(pts[:,0], [0,.25,.5,.75,1.000001], labels=['L','M','R'])
        print(name, "quadrant-x counts:", q.value_counts().sort_index().to_dict())

    # flow lengths and cross distances
    flow_len = np.hypot(o[:,0]-d[:,0], o[:,1]-d[:,1])
    print("\nflow length: min/mean/max = %.3f / %.3f / %.3f" % (flow_len.min(), flow_len.mean(), flow_len.max()))
    print("share of flows with len < 0.3:", (flow_len<0.3).mean().round(3), " (uniform expectation ~ 1-? )")

    # quick look at flow-to-flow distance (max of endpoint distances) - small scale
    do = cdist(o,o)
    dd = cdist(d,d)
    ddist = np.maximum(do, dd)
    np.fill_diagonal(ddist, np.inf)
    nn = ddist.min(axis=1)
    print("nearest-neighbor flow distance: mean %.4f median %.4f (CSR approx 0.05*?)" % (nn.mean(), np.median(nn)))
    for r in [0.05,0.1,0.2,0.3,0.5]:
        print("  share of flows with a neighbor within r=%.2f: %.3f" % (r, (ddist<=r).any(axis=1).mean()))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 3 ---

for name, pts in [("origins", o), ("dests", d)]:
    qx = pd.cut(pts[:,0], [0,.25,.5,.75,1.000001], labels=['Q1','Q2','Q3','Q4'])
    qy = pd.cut(pts[:,1], [0,.25,.5,.75,1.000001], labels=['B','M','T','T2'])
    print(name, "x-quadrant counts:", qx.value_counts().sort_index().to_dict())

flow_len = np.hypot(o[:,0]-d[:,0], o[:,1]-d[:,1])
print("\nflow length: min/mean/max = %.3f / %.3f / %.3f" % (flow_len.min(), flow_len.mean(), flow_len.max()))

do = cdist(o,o)
dd = cdist(d,d)
ddist = np.maximum(do, dd)
np.fill_diagonal(ddist, np.inf)
nn = ddist.min(axis=1)
print("nearest-neighbor flow distance: mean %.4f median %.4f" % (nn.mean(), np.median(nn)))
for r in [0.05,0.1,0.2,0.3,0.5]:
    print("  share of flows with a neighbor within r=%.2f: %.3f" % (r, (ddist<=r).any(axis=1).mean()))

# --- block 4 (wrapped in try/except) ---
try:

    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.geometry import Polygon
    import os, time

    np.random.seed(2024)

    # ---------- geometry ----------
    o = flow[['x_o','y_o']].values
    d = flow[['x_d','y_d']].values
    o_sq = Polygon([(0,0),(0,1),(1,1),(1,0)])
    d_sq = Polygon([(0,0),(0,1),(1,1),(1,0)])

    def poly_area(g):
        try:
            return float(g.area)
        except TypeError:
            return sum(p.area for p in g.geoms)

    # ---------- distance grid ----------
    n_r = 121
    r = np.linspace(0.005, 1.0, n_r)
    r2 = r*r

    def P_matrix(O, D, r):
        # |B(o,r)∩O| / A_O  and  |B(dd,r)∩D| / A_D  for every flow
        po = np.empty(n_r); pd_ = np.empty(n_r)
        for t, rv in enumerate(r):
            po[t] = sum(poly_area(o_sq.intersection(o[i].tolist() and __import__('shapely').geometry.Point(o[i]).buffer(rv))) for i in range(len(O)))/o_sq.area
            pd_[t] = sum(poly_area(d_sq.intersection(__import__('shapely').geometry.Point(D[i]).buffer(rv))) for i in range(len(D)))/d_sq.area
        return po[:,None], pd_[:,None]

    print("computing edge correction P(i,r) ...")
    t0 = time.time()
    Po, Pd = P_matrix(o, d, r)     # (n_r, n) each
    print("done in %.1f s" % (time.time()-t0))

    # ---------- flow-to-flow distances ----------
    import scipy.spatial.distance as sdist
    Do = sdist.cdist(o,o)
    Dd = sdist.cdist(d,d)
    DD = np.maximum(Do, Dd)         # flow distance d_ij
    np.fill_diagonal(DD, np.inf)

    # ---------- local K per flow (row i of the product distance matrix) ----------
    # N_i(t) = number of j with DD_ij <= r_t  (vector over i)
    N = (DD <= r[None,:]).sum(axis=0)          # (n, n_r)
    K_local = (N.T / 1000) / (Po/Po)  # placeholder, overwritten below
    K_local = (N.T) / (1000 * (Po*Pd).T)   # (n, n_r): N_i / (n * P_i)
    K_local = np.pi * r2[None,:] * (N.T) / (1000 * (Po*Pd).T)
    K = K_local.mean(axis=0)
    L = np.sqrt(np.maximum(K/np.pi, 0)) - r
    print("K range:", K.min().round(4), K.max().round(4))
    print("L at r=0.05,0.1,0.2,0.5,1.0:", L[9].round(3), L[19].round(3), L[39].round(3), L[99].round(3), L[-1].round(3))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 5 (wrapped in try/except) ---
try:

    from shapely.geometry import Point
    import numpy as np

    def P_matrix(O, D, win_o, win_d, r, n_r):
        n = len(O)
        po = np.empty((n_r, n)); pd_ = np.empty((n_r, n))
        Pt_o = [Point(*pt) for pt in O]
        Pt_d = [Point(*pt) for pt in D]
        for t, rv in enumerate(r):
            po[t] = np.array([poly_area(win_o.intersection(p.buffer(rv))) for p in Pt_o]) / win_o.area
            pd_[t] = np.array([poly_area(win_d.intersection(p.buffer(rv))) for p in Pt_d]) / win_d.area
        return po, pd_

    import time
    t0 = time.time()
    Po, Pd = P_matrix(o, d, o_sq, d_sq, r, n_r)   # (n_r, n) each, clipped/normalized
    print("P done %.1fs; P ranges: Po %.3f-%.3f, Pd %.3f-%.3f" % (
        time.time()-t0, Po.min(), Po.max(), Pd.min(), Pd.max()))

    n = len(o)
    P = Po * Pd                        # (n_r, n) product-window edge correction
    N = (DD <= r[:,None]).sum(axis=1)  # (n_r, n) neighbor counts
    K_local = np.pi * r2[:,None] * N / (n * P)
    K = K_local.mean(axis=1)
    L = np.sqrt(np.maximum(K/np.pi, 0)) - r
    np.save('pred_results/K_local.npy', K_local)  # placeholder dir may not exist yet; ignore if fails
    print("K range: %.4f .. %.4f" % (K.min(), K.max()))
    print("r   : %.3f  %.3f  %.3f  %.3f  %.3f" % (r[9], r[19], r[39], r[99], r[-1]))
    print("L(r): % .3f % .3f % .3f % .3f % .3f" % (L[9], L[19], L[39], L[99], L[-1]))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 6 (wrapped in try/except) ---
try:

    import numpy as np, os
    os.makedirs('pred_results', exist_ok=True)

    N = (DD[None,:,:] <= r[:,None,:]).sum(axis=2)   # (n_r, n)
    K_local = np.pi * r2[:,None] * N / (n * P)      # (n_r, n)
    K = K_local.mean(axis=1)
    L = np.sqrt(np.maximum(K/np.pi, 0)) - r

    np.save('pred_results/K_local.npy', K_local)
    print("K range: %.4f .. %.4f" % (K.min(), K.max()))
    print("L(r) at selected r:")
    for idx in [9,19,39,59,99,120]:
        print("  r=%.3f  K=%.3f  L=%+.3f" % (r[idx], K[idx], L[idx]))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 7 ---

import numpy as np
N = (DD[None,:,:] <= r[:,None,None]).sum(axis=2)   # (n_r, n)
K_local = np.pi * r2[:,None] * N / (n * P)         # (n_r, n)
K = K_local.mean(axis=1)
L = np.sqrt(np.maximum(K/np.pi, 0)) - r
np.save('pred_results/K_local.npy', K_local)
print("N range:", N.min(), N.max())
print("K range: %.4f .. %.4f" % (K.min(), K.max()))
for idx in [9,19,39,59,99,120]:
    print("  r=%.3f  K=%.3f  L=%+.3f" % (r[idx], K[idx], L[idx]))

# --- block 8 (wrapped in try/except) ---
try:

    import numpy as np, time
    rng = np.random.default_rng(2024)
    n_sims = 99
    Lsim = np.empty((n_sims, n_r))
    t0 = time.time()
    batch = 25   # r-values per memory block
    for s in range(n_sims):
        os_ = rng.random((n,2)); ds_ = rng.random((n,2))
        Dos = sdist.cdist(os_, os_); Dds = sdist.cdist(ds_, ds_)
        Ds_ = np.maximum(Dos, Dds)
        for b0 in range(0, n_r, batch):
            b = slice(b0, min(b0+batch, n_r))
            Nb = (Ds_[None,:,:] <= r[b,None,None]).sum(axis=2)
            Ks = (np.pi * r2[b,None] * Nb) / (n * P[b])
            Lsim[s,b] = np.sqrt(np.maximum(Ks/np.pi, 0)) - r[b]
        if s % 20 == 0:
            print("sim %d/%d  (%.1fs)" % (s+1, n_sims, time.time()-t0))
    lo, hi = np.percentile(Lsim, [2.5, 97.5], axis=0)
    np.save('pred_results/L_sim_envelope.npy', np.stack([lo, hi]))
    print("done %.1fs" % (time.time()-t0))
    print("envelope width at r=0.163: ±%.3f ; at r=0.328: %.3f..%.3f ; at r=0.826: %.3f..%.3f" %
          ((lo[19], hi[19]), lo[39], hi[39], lo[99], hi[99]))
    print("mean envelope: %.3f .. %.3f" % (lo.mean(), hi.mean()))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 9 (wrapped in try/except) ---
try:

    import numpy as np, time
    rng = np.random.default_rng(2024)
    n_sims = 99
    Lsim = np.empty((n_sims, n_r))
    t0 = time.time()
    for s in range(n_sims):
        os_ = rng.random((n,2)); ds_ = rng.random((n,2))
        Ds_ = np.maximum(sdist.cdist(os_, os_), sdist.cdist(ds_, ds_))
        for t in range(n_r):
            Nt = (Ds_ <= r[t]).sum(axis=1)
            Kt = np.pi * r2[t] * Nt / (n * P[t])
            Lsim[s,t] = np.sqrt(np.maximum(Kt/np.pi, 0)) - r[t]
        if s % 25 == 0:
            print("sim %d/%d  (%.1fs)" % (s+1, n_sims, time.time()-t0))
    lo, hi = np.percentile(Lsim, [2.5, 97.5], axis=0)
    np.save('pred_results/L_sim_envelope.npy', np.stack([lo, hi]))
    print("done %.1fs" % (time.time()-t0))
    for idx in [9,19,39,59,99,120]:
        print("  r=%.3f  L_obs=%+.4f  CI95=[%+.4f, %+.4f]" % (r[idx], L[idx], lo[idx], hi[idx]))
    print("envelope midpoints ~0? mean of (lo+hi)/2 =", (lo+hi)/2).mean().round(4)
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 10 (wrapped in try/except) ---
try:

    import numpy as np, time
    rng = np.random.default_rng(2024)
    n_sims = 99
    Lsim = np.empty((n_sims, n_r))
    t0 = time.time()
    for s in range(n_sims):
        os_ = rng.random((n,2)); ds_ = rng.random((n,2))
        Ds_ = np.maximum(sdist.cdist(os_, os_), sdist.cdist(ds_, ds_))
        for t in range(n_r):
            Kbar = np.pi * r2[t] * (Ds_ <= r[t]).sum(axis=1).mean() / (n * P[t])
            Lsim[s,t] = np.sqrt(np.maximum(Kbar/np.pi, 0)) - r[t]
        if s % 25 == 0:
            print("sim %d/%d  (%.1fs)" % (s+1, n_sims, time.time()-t0))
    lo, hi = np.percentile(Lsim, [2.5, 97.5], axis=0)
    np.save('pred_results/L_sim_envelope.npy', np.stack([lo, hi]))
    print("done %.1fs" % (time.time()-t0))
    for idx in [9,19,39,59,99,120]:
        print("  r=%.3f  L_obs=%+.4f  CI95=[%+.4f, %+.4f]" % (r[idx], L[idx], lo[idx], hi[idx]))
    print("envelope centered on 0? mean of midpoints =", round(float((lo+hi)/2).mean(), 4))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 11 ---

import shapely, time
pts = shapely.points(o)
t0=time.time()
buf = shapely.buffer(pts, 0.2)
area = shapely.area(shapely.intersection(buf, o_sq))
print("vectorized 1000 buffers+intersection: %.2f ms" % ((time.time()-t0)*1000))
print("area stats:", area.mean().round(4), area.min().round(4), area.max().round(4))
# analytic check: r=0.2 near center -> ~pi*0.04=0.1257; near corner smaller
print("shapely version:", shapely.__version__)

# --- block 12: omitted (its state is recomputed by later blocks) ---

# --- block 13 (wrapped in try/except) ---
try:

    import numpy as np, shapely

    def H_quad(x, y, R):
        """Area within [0,x]x[0,y] (x,y>=0) of the disk of radius R centered at origin."""
        x = np.asarray(x, float); y = np.asarray(y, float)
        H = np.empty(np.broadcast(x, y).shape)
        R = float(R)
        xR = np.minimum(x, R); yR = np.minimum(y, R)
        # case A: x>=R, y>=R
        H = np.pi*R*R/4.0
        m1 = (x < R) | (y < R)
        # case B: x>=R, y<R
        m2 = m1 & (x >= R)
        yb = np.where(m2, y, 1.0)
        H = np.where(m2, yb*np.sqrt(R*R - yb*yb)/2 + (R*R/2)*np.arcsin(yb/R), H)
        # case C: y>=R, x<R
        m3 = m1 & ~(x >= R) & (y >= R)
        xc = np.where(m3, x, 1.0)
        H = np.where(m3, xc*np.sqrt(R*R - xc*xc)/2 + (R*R/2)*np.arcsin(xc/R), H)
        # case D: x<R, y<R
        m4 = m1 & ~(x >= R) & ~(y >= R)
        xd = np.where(m4, x, 1.0); yd = np.where(m4, y, 1.0)
        inside = xd*xd + yd*yd <= R*R
        Hd = xd*yd  # default
        t = np.sqrt(R*R - yd*yd)  # t*
        alt = (yd*t)/2 + xd*np.sqrt(R*R - xd*xd)/2 + (R*R/2)*(np.arcsin(xd/R) + np.arcsin(yd/R) - np.pi/2)
        H = np.where(m4, np.where(inside, xd*yd, alt), H)
        return H

    def circ_rect_area(cx, cy, R, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
        """Exact area of disk(center=(cx,cy), R) intersected with rect [x0,x1]x[y0,y1]."""
        cx = np.asarray(cx, float); cy = np.asarray(cy, float)
        def signH(a, b):
            return np.sign(a)*np.sign(b)*H_quad(np.abs(a), np.abs(b), R)
        A = ( signH(x1-cx, y1-cy) - signH(x0-cx, y1-cy)
             - signH(x1-cx, y0-cy) + signH(x0-cx, y0-cy) )
        return np.maximum(A, 0.0)

    # --- verify against shapely ---
    rng = np.random.default_rng(1)
    pts = rng.random((200,2)); Rs = rng.uniform(0.02, 1.2, 200)
    ref = shapely.area(shapely.intersection(shapely.buffer(shapely.points(pts), Rs), o_sq))
    num = circ_rect_area(pts[:,0], pts[:,1], Rs)
    err = np.abs(num - ref)
    print("max abs err vs shapely: %.3e ; mean err: %.3e" % (err.max(), err.mean()))
    assert err.max() < 1e-10, "analytic formula mismatch"
    print("analytic edge correction verified OK")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 14 (wrapped in try/except) ---
try:

    import numpy as np, shapely

    def H_quad(x, y, R):
        """Area within [0,x]x[0,y] (x,y>=0) of disk radius R centered at origin.
           x, y, R: scalars or same-shape arrays."""
        x = np.asarray(x, float); y = np.asarray(y, float); R = np.asarray(R, float)
        x, y, R = np.broadcast_arrays(x, y, R)
        H = np.pi*R*R/4.0
        m1 = (x < R) | (y < R)
        m2 = m1 & (x >= R)
        yb = np.where(m2, y, 1.0)
        H = np.where(m2, yb*np.sqrt(np.maximum(R*R - yb*yb, 0))/2 + (R*R/2)*np.arcsin(np.minimum(yb/R, 1)), H)
        m3 = m1 & ~(x >= R) & (y >= R)
        xc = np.where(m3, x, 1.0)
        H = np.where(m3, xc*np.sqrt(np.maximum(R*R - xc*xc, 0))/2 + (R*R/2)*np.arcsin(np.minimum(xc/R, 1)), H)
        m4 = m1 & ~(x >= R) & ~(y >= R)
        xd = np.where(m4, x, 1.0); yd = np.where(m4, y, 1.0)
        inside = xd*xd + yd*yd <= R*R
        t = np.sqrt(np.maximum(R*R - yd*yd, 0))
        alt = (yd*t)/2 + xd*np.sqrt(np.maximum(R*R - xd*xd, 0))/2 + (R*R/2)*(np.arcsin(np.minimum(xd/R,1)) + np.arcsin(np.minimum(yd/R,1)) - np.pi/2)
        H = np.where(m4, np.where(inside, xd*yd, alt), H)
        return H

    def circ_rect_area(cx, cy, R, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
        cx = np.asarray(cx, float); cy = np.asarray(cy, float); R = np.asarray(R, float)
        def signH(a, b):
            return np.sign(a)*np.sign(b)*H_quad(np.abs(a), np.abs(b), R)
        A = ( signH(x1-cx, y1-cy) - signH(x0-cx, y1-cy)
             - signH(x1-cx, y0-cy) + signH(x0-cx, y0-cy) )
        return np.maximum(A, 0.0)

    rng = np.random.default_rng(1)
    pts = rng.random((200,2)); Rs = rng.uniform(0.02, 1.2, 200)
    ref = shapely.area(shapely.intersection(shapely.buffer(shapely.points(pts), Rs), o_sq))
    num = circ_rect_area(pts[:,0], pts[:,1], Rs)
    err = np.abs(num - ref)
    print("max abs err vs shapely: %.3e ; mean: %.3e" % (err.max(), err.mean()))
    assert err.max() < 1e-10
    # also test scalar R with array points
    ref2 = shapely.area(shapely.intersection(shapely.buffer(shapely.points(pts), 0.3), o_sq))
    num2 = circ_rect_area(pts[:,0], pts[:,1], 0.3)
    print("scalar-R max err: %.3e" % np.abs(num2-ref2).max())
    assert np.abs(num2-ref2).max() < 1e-10
    print("analytic edge correction verified OK (both per-point and shared radii)")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 15 (wrapped in try/except) ---
try:

    import numpy as np, shapely
    rng = np.random.default_rng(1)
    pts = rng.random((200,2)); Rs = rng.uniform(0.02, 1.2, 200)
    ref = shapely.area(shapely.intersection(shapely.buffer(shapely.points(pts), Rs, quad_segs=256), o_sq))
    num = circ_rect_area(pts[:,0], pts[:,1], Rs)
    err = np.abs(num - ref)
    print("high-res ref: max abs err %.3e, mean %.3e" % (err.max(), err.mean()))
    ref2 = shapely.area(shapely.intersection(shapely.buffer(shapely.points(pts), 0.3, quad_segs=256), o_sq))
    num2 = circ_rect_area(pts[:,0], pts[:,1], 0.3)
    print("scalar-R max err: %.3e" % np.abs(num2-ref2).max())
    # extreme cases: center, corners, huge R
    for cx, cy, Rr in [(0.5,0.5,1.5),(0,0,0.4),(1,1,0.3),(0.2,0.8,0.9)]:
        rr = shapely.area(shapely.intersection(shapely.buffer(shapely.Point(cx,cy), Rr, quad_segs=512), o_sq))
        nn = float(circ_rect_area(np.array([cx]), np.array([cy]), np.array([Rr]))[0])
        print("  c=(%.1f,%.1f) R=%.1f  shapely=%.8f  analytic=%.8f  err=%.2e" % (cx,cy,Rr,rr,nn,abs(nn-rr)))
    assert err.max() < 1e-9
    print("analytic edge correction verified OK")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 16 ---

import numpy as np
# exact per-flow edge correction for observed data (windows = unit squares, area=1)
Ao = np.empty((n, n_r)); Ad = np.empty((n, n_r))
for t in range(n_r):
    Ao[:,t] = circ_rect_area(o[:,0], o[:,1], r[t])
    Ad[:,t] = circ_rect_area(d[:,0], d[:,1], r[t])
Pobs = (Ao/1.0)*(Ad/1.0)           # (n, n_r)
PobsT = Pobs.T                      # (n_r, n)
# compare with old shapely-based P (which is (n_r, n))
print("Pobs vs old shapely P: max abs diff = %.3e" % np.abs(PobsT - P).max())
print("Pobs range: %.4f .. %.4f ; at r=0.2: mean %.4f" % (PobsT.min(), PobsT.max(), PobsT[39].mean()))

# recompute observed K, L with exact P
N = (DD[None,:,:] <= r[:,None,None]).sum(axis=2)   # (n_r, n)
K_local = np.pi * r2[:,None] * N / (n * PobsT)     # (n_r, n)
K = K_local.mean(axis=1)
L = np.sqrt(np.maximum(K/np.pi, 0)) - r
np.save('pred_results/K_local.npy', K_local)
print("observed K range: %.4f .. %.4f" % (K.min(), K.max()))
for idx in [9,19,39,59,99,120]:
    print("  r=%.3f  K=%.3f  L=%+.4f" % (r[idx], K[idx], L[idx]))

# --- block 17 (wrapped in try/except) ---
try:

    import numpy as np, time
    rng = np.random.default_rng(2024)
    n_sims = 99
    Lsim = np.empty((n_sims, n_r))
    t0 = time.time()
    for s in range(n_sims):
        os_ = rng.random((n,2)); ds_ = rng.random((n,2))
        ao = np.empty((n,n_r)); ad = np.empty((n,n_r))
        for t in range(n_r):
            ao[:,t] = circ_rect_area(os_[:,0], os_[:,1], r[t])
            ad[:,t] = circ_rect_area(ds_[:,0], ds_[:,1], r[t])
        Ps = ao*ad                       # (n, n_r) product-window correction per simulated flow
        Ds_ = np.maximum(sdist.cdist(os_, os_), sdist.cdist(ds_, ds_))
        for t in range(n_r):
            Kbar = np.pi * r2[t] * (Ds_ <= r[t]).sum(axis=1).mean() / (n * Ps[:,t].mean())
            Lsim[s,t] = np.sqrt(np.maximum(Kbar/np.pi, 0)) - r[t]
        if s % 25 == 0:
            print("sim %d/%d (%.1fs)" % (s+1, n_sims, time.time()-t0))
    lo, hi = np.percentile(Lsim, [2.5, 97.5], axis=0)
    np.save('pred_results/L_sim_envelope.npy', np.stack([lo, hi]))
    print("done %.1fs" % (time.time()-t0))
    for idx in [9,19,39,59,99,120]:
        print("  r=%.3f  L_obs=%+.4f  CI95=[%+.4f, %+.4f]  out=%s" % (
            r[idx], L[idx], lo[idx], hi[idx], not (lo[idx] < L[idx] < hi[idx])))
    print("envelope midpoints mean: %.4f" % float((lo+hi)/2).mean())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 18 (wrapped in try/except) ---
try:

    import numpy as np, time
    rng = np.random.default_rng(2024)
    n_sims = 99
    Lsim = np.empty((n_sims, n_r))
    t0 = time.time()
    for s in range(n_sims):
        os_ = rng.random((n,2)); ds_ = rng.random((n,2))
        ao = np.empty((n,n_r)); ad = np.empty((n,n_r))
        for t in range(n_r):
            ao[:,t] = circ_rect_area(os_[:,0], os_[:,1], r[t])
            ad[:,t] = circ_rect_area(ds_[:,0], ds_[:,1], r[t])
        Ps = ao*ad
        Ds_ = np.maximum(sdist.cdist(os_, os_), sdist.cdist(ds_, ds_))
        np.fill_diagonal(Ds_, np.inf)          # exclude self-pair
        for t in range(n_r):
            Kbar = np.pi * r2[t] * (Ds_ <= r[t]).sum(axis=1).mean() / (n * Ps[:,t].mean())
            Lsim[s,t] = np.sqrt(np.maximum(Kbar/np.pi, 0)) - r[t]
        if s % 25 == 0:
            print("sim %d/%d (%.1fs)" % (s+1, n_sims, time.time()-t0))
    lo, hi = np.percentile(Lsim, [2.5, 97.5], axis=0)
    np.save('pred_results/L_sim_envelope.npy', np.stack([lo, hi]))
    print("done %.1fs" % (time.time()-t0))
    for idx in [0,9,19,39,59,99,120]:
        print("  r=%.3f  L_obs=%+.4f  CI95=[%+.4f, %+.4f]" % (r[idx], L[idx], lo[idx], hi[idx]))
    print("midpoint mean: %.4f (should be ~0)" % float((lo+hi)/2).mean())
    print("Lsim range: %.3f .. %.3f" % (Lsim.min(), Lsim.max()))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 19 ---

import numpy as np, pandas as pd
env = np.load('pred_results/L_sim_envelope.npy')   # (2, n_r): lo, hi
lo, hi = env[0], env[1]
print("envelope midpoint mean: %.5f" % float(np.mean((lo+hi)/2)))

# --- critical distances: zero crossings of L(r) via linear interpolation ---
cross = []
for i in range(len(r)-1):
    if L[i] == 0: cross.append(r[i]); continue
    if L[i]*L[i+1] < 0:
        rc = r[i] - L[i]*(r[i+1]-r[i])/(L[i+1]-L[i])
        cross.append(rc)
print("zero-crossings (critical distances):", [round(c,4) for c in cross])

# --- deliverable 1: CSV ---
out = pd.DataFrame({
    'r': r,
    'K': K,
    'L': L,
    'CI_lo': lo,
    'CI_hi': hi,
    'pattern': np.where(L > 0, 'clustering', np.where(L < 0, 'dispersion', 'neutral')),
    'outside_95CI': (L < lo) | (L > hi),
})
out.to_csv('pred_results/L_function.csv', index=False, float_format='%.6f')
print(out.head(3).to_string())
print("rows:", len(out), "  outside CI: %d / %d" % (out['outside_95CI'].sum(), len(out)))

# --- block 20 (wrapped in try/except) ---
try:

    import numpy as np, pandas as pd
    out['outside_95CI'] = (L < lo - 1e-12) | (L > hi + 1e-12)
    out.to_csv('pred_results/L_function.csv', index=False, float_format='%.6f')
    print("rows:", len(out), "| outside 95% CI: %d / %d" % (out['outside_95CI'].sum(), len(out)))
    print("pattern along r:")
    for idx in [9,19,39,59,79,99,120]:
        print("  r=%.3f  L=%+.4f  CI=[%.4f, %.4f]  %s" % (r[idx], L[idx], lo[idx], hi[idx], out['pattern'][idx]))
    print("critical distances (zero crossings of L):", [round(float(c),4) for c in cross])
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 21 ---

import numpy as np, pandas as pd
out['outside_95CI'] = (L < lo - 1e-12) | (L > hi + 1e-12)
out.to_csv('pred_results/L_function.csv', index=False, float_format='%.6f')
print("rows:", len(out), "| outside 95 pct CI: %d / %d" % (out['outside_95CI'].sum(), len(out)))
print("pattern along r:")
for idx in [9,19,39,59,79,99,120]:
    print("  r=%.3f  L=%+.4f  CI=[%.4f, %.4f]  %s" % (r[idx], L[idx], lo[idx], hi[idx], out['pattern'][idx]))
print("critical distances (zero crossings of L):", [round(float(c),4) for c in cross])

# --- block 22 ---

import numpy as np
print(out[out['outside_95CI']].to_string())
i = out['outside_95CI'].idxmax()
print("i =", i, "r=%.5f L=%.8f lo=%.8f hi=%.8f" % (r[i], L[i], lo[i], hi[i]))

# --- block 23 ---

import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

fig = plt.figure(figsize=(11.5, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.55], wspace=0.24)

# ---------- panel 1: flow map ----------
ax1 = fig.add_subplot(gs[0])
step = 3
for i in range(0, len(o), step):
    ax1.plot([o[i,0], d[i,0]], [o[i,1], d[i,1]], color='steelblue', alpha=0.10, lw=0.5, zorder=1)
ax1.scatter(o[:,0], o[:,1], s=6, c='tab:blue', alpha=0.55, label='origins (n=1000)', zorder=3)
ax1.scatter(d[:,0], d[:,1], s=6, c='tab:orange', alpha=0.55, label='destinations (n=1000)', zorder=3)
for sq, c, lab in [(o_sq, 'tab:blue', 'origin window (1x1)'), (d_sq, 'tab:orange', 'destination window (1x1)')]:
    xs, ys = sq.exterior.xy
    ax1.plot(xs, ys, color=c, ls='--', lw=1.2, label=lab)
ax1.set_xlim(-0.05, 1.05); ax1.set_ylim(-0.05, 1.05); ax1.set_aspect('equal')
ax1.set_title('Geographical flows (OD pairs) and analysis windows')
ax1.legend(fontsize=7, loc='upper left'); ax1.set_xlabel('x'); ax1.set_ylabel('y')

# ---------- panel 2: L-function ----------
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(r, lo, hi, color='lightgray', alpha=0.85, lw=0, label='95% Monte-Carlo envelope (99 CSR sims)')
ax2.plot(r, r, 'k--', lw=1, label='CSR reference (L = r ... L=0 line)')
ax2.axhline(0, color='k', lw=0.7)
# shade clustering / dispersion bands of the observed L
ax2.fill_between(r, 0, L, where=L>0, color='tab:red', alpha=0.15, label='L(r) > 0: clustering')
ax2.fill_between(r, 0, L, where=L<0, color='tab:blue', alpha=0.15, label='L(r) < 0: dispersion')
ax2.plot(r, L, color='darkred', lw=1.6, label='observed L(r)')

# critical distances (zero crossings), skip the degenerate r~0.007 one
crit = [c for c in cross if c > 0.02]
labels = {0.04698587866638293: 'r1 = 0.047\nrepulsion to clustering',
          0.08657873342486114: 'r2 = 0.087\nclustering band start',
          0.432191612792105: 'r3 = 0.432\nclustering band end',
          0.87230087638: 'r4 = 0.872\nrepulsion ends'}
for c in crit:
    lc = labels.get(round(float(c),6), None)
    key = [k for k in labels if abs(k-c) < 1e-3]
    if key:
        ax2.axvline(c, color='gray', ls=':', lw=1.0)
        yy = np.interp(c, r, L)
        ax2.annotate(labels[key[0]], xy=(c, yy), xytext=(c+0.015, yy+0.012),
                     fontsize=6.6, ha='left', arrowprops=dict(arrowstyle='->', lw=0.6, color='gray'))
ax2.plot([0.146], [L[np.argmin(abs(r-0.146))]], 'rv', ms=6)
ax2.annotate('marginal clustering\n(L > upper envelope)', xy=(0.146, 0.0037), xytext=(0.19, 0.02),
             fontsize=6.8, arrowprops=dict(arrowstyle='->', lw=0.7, color='red'))

ax2.set_xlabel('distance r'); ax2.set_ylabel('L(r)')
ax2.set_title('Global L-function of the flow (mean of local K-functions, edge-corrected)')
ax2.set_xlim(0, 1.0); ax2.set_ylim(-0.03, 0.035)
ax2.legend(fontsize=6.8, loc='lower right')
ax2.grid(alpha=0.25)

fig.suptitle('Spatial aggregation of geographical flows via the L-function (n = 1000 OD pairs)', fontsize=11, y=0.995)
plt.savefig('pred_results/L_function.png', dpi=150, bbox_inches='tight')
plt.close()
print("figure saved")

# --- block 24 ---

import pandas as pd, numpy as np, os
os.makedirs('pred_results', exist_ok=True)
f = 'pred_results/L_function.csv'
df = pd.read_csv(f)
print("files in pred_results/:", sorted(os.listdir('pred_results/')))
print("PNG size: %.0f KB" % (os.path.getsize('pred_results/L_function.png')/1024))
print()
print("CSV: %d rows, cols=%s" % (len(df), list(df.columns)))
print("r range: %.3f .. %.3f" % (df.r.min(), df.r.max()))
print("K range: %.4f .. %.4f  (varies: %s)" % (df.K.min(), df.K.max(), df.K.nunique() > 50))
print("L range: %+.4f .. %+.4f" % (df.L.min(), df.L.max()))
print("CI width range: %.4f .. %.4f" % ((df.CI_hi-df.CI_lo).min(), (df.CI_hi-df.CI_lo).max()))
print("patterns:", df.pattern.value_counts().to_dict())
print("rows outside 95 pct CI: %d (at r=%.3f, L=%+.4f > hi=%+.4f)" % (
    df.outside_95CI.sum(), df.loc[df.outside_95CI,'r'].iloc[0],
    df.loc[df.outside_95CI,'L'].iloc[0], df.loc[df.outside_95CI,'CI_hi'].iloc[0]))
print()
print("Critical distances (zero crossings of L): 0.047, 0.087, 0.432, 0.444, 0.445, 0.872")
print("Interpretation: near-field repulsion (r<0.05), weak mid-range clustering (0.09-0.43),")
print("long-range dispersion (0.43-0.87) -> flow is close to CSR; no systematic aggregation scale.")
