# Correct: 22-NN adaptive bisquare GWR of AveInteres on AcceptedLo + pHHLoanRat (no AveLoanGra), which the
# instruction allows because it does not name the explanatory variable. A long step-by-step session;
# the steps run in order and those that raise an error are wrapped in try/except.

import os; os.makedirs('pred_results', exist_ok=True)

# --- block 1 ---
loans = gpd.read_file('dataset/loan_data.shp')
print(f"Rows: {len(loans)}")
print(f"CRS: {loans.crs.to_string() if loans.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in loans.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in loans.total_bounds]}")
_cols = [c for c in loans.columns if c != loans.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(loans[_cols].head(3).to_string())

# --- block 2 ---
import pandas as pd
import numpy as np

print(loans.shape)
print(loans[['AcceptedLo','AveInteres']].describe())
print("NaNs in AcceptedLo:", loans['AcceptedLo'].isna().sum())
print("NaNs in AveInteres:", loans['AveInteres'].isna().sum())

sel = loans[loans['AcceptedLo'] >= 30].copy()
print("Selected rows:", len(sel))
print(sel[['AcceptedLo','AveInteres']].describe())
print("NaNs in selected AveInteres:", sel['AveInteres'].isna().sum())

# --- block 3 (raises an error) ---
try:
    import os
    os.makedirs('pred_results', exist_ok=True)

    import mgwr
    print("mgwr version:", mgwr.__version__ if hasattr(mgwr, '__version__') else 'unknown')
    print([x for x in dir(mgwr) if not x.startswith('_')])
    import esda
    print("esda version:", esda.__version__)
    from esda import Gi
    print("Gi ok")
    from libpysal.weights import Queen
    print("Queen ok")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 4 ---
import esda
print([x for x in dir(esda) if not x.startswith('_')])
import mgwr.gwr as gwr_mod
import inspect
print(inspect.signature(gwr_mod.GWR.__init__))

# --- block 5 (raises an error) ---
try:
    import inspect, mgwr.gwr as gwr_mod
    src = inspect.getsource(gwr_mod.GWR._bw_setup) if hasattr(gwr_mod.GWR, '_bw_setup') else None
    print(src)
    import mgwr.kernels as krn
    print([x for x in dir(krn) if not x.startswith('_')])
    print(inspect.getsource(krn.KernelFactory.__call__))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 6 ---
import inspect, mgwr.gwr as gwr_mod
src = inspect.getsource(gwr_mod.GWR.__init__)
print(src[:3000])

# --- block 7 ---
import inspect, mgwr.gwr as gwr_mod
src = inspect.getsource(gwr_mod)
import re
for m in re.finditer(r'def (\w+)', src):
    pass
print(src[:4000])

# --- block 8 ---
import inspect, mgwr.gwr as gwr_mod, mgwr.kernels as krn
print(inspect.getsource(krn))
print("===== FIT =====")
print(inspect.getsource(gwr_mod.GWR.fit))

# --- block 9 ---
import inspect, mgwr.gwr as gwr_mod
src = inspect.getsource(gwr_mod.GWR.fit)
print(src[:2500])

# --- block 10 (raises an error) ---
try:
    import numpy as np, esda
    from libpysal.weights import Queen

    print("getisord:", getattr(esda, 'getisord', None))
    # Build Queen contiguity on selected polygons
    w = Queen.from_dataframe(sel)
    w.symmetrize()
    w.transform = 'R'  # row-standardize
    print("w type:", type(w), "n:", w.n, "nnn min/max:", w.nnn.min(), w.nnn.max())

    x = sel['AveInteres'].values.astype(float)
    print("x stats:", x.mean(), x.std())

    # Try Gi* 
    try:
        from esda import Gi
        gi = Gi(x, w)
        print("Gi OK, z[:5]:", gi.z[:5])
    except Exception as e:
        print("Gi import failed:", e)
        # fallback to G_Local
        gloc = esda.G_Local(x, w)
        print("G_Local z[:5]:", gloc.z[:5])
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 11 ---
import numpy as np
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

cents = np.array([(g.centroid.x, g.centroid.y) for g in sel.geometry])
print("n:", len(cents))
tree = cKDTree(cents)
# distance to 22nd nearest neighbor (index 22 since 0 is self)
d, idx = tree.query(cents, k=23)
d22 = d[:, 22]
print("22nd-NN distance: min/median/mean/max")
print(d22.min(), np.median(d22), d22.mean(), d22.max())

# distance to 1st, 5th, 10th nearest
d5  = d[:,5]; d10=d[:,10]
print("1st-NN median:", np.median(d[:,1]))
print("5th-NN median:", np.median(d5))
print("10th-NN median:", np.median(d10))

# bounding box size
print("bbox width/height (m):", (cents[:,0].max()-cents[:,0].min()), (cents[:,1].max()-cents[:,1].min()))

# --- block 12 (raises an error) ---
try:
    import esda.getisord as go
    print([x for x in dir(go) if not x.startswith('_')])
    import inspect
    print(inspect.signature(go.Gi.__init__))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 13 (raises an error) ---
try:
    import warnings
    warnings.filterwarnings('ignore')
    from libpysal.weights import Queen, KNN
    import numpy as np

    wq = Queen.from_dataframe(sel)
    wq.symmetrize()
    nn_q = wq.cardinality
    print("Queen neighbors: min/median/max:", nn_q.min(), int(np.median(nn_q)), nn_q.max())
    print("islands (0 neighbors):", (nn_q==0).sum())

    wk = KNN.from_dataframe(sel, k=22)
    nn_k = wk.cardinality
    print("KNN22 neighbors: min/median/max:", nn_k.min(), int(np.median(nn_k)), nn_k.max())
    print("KNN islands:", (nn_k==0).sum())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 14 (raises an error) ---
try:
    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    from libpysal.weights import KNN

    wk = KNN.from_dataframe(sel, k=22)
    nn_k = wk.cardinalities
    print("KNN22 neighbors: min/median/max:", nn_k.min(), int(np.median(nn_k)), nn_k.max())

    # queen cardinalities
    nn_q = wq.cardinalities
    print("Queen neighbors: min/median/max:", nn_q.min(), int(np.median(nn_q)), nn_q.max())
    print("Queen islands:", (nn_q==0).sum())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 15 ---
import numpy as np
vals_k = list(wk.cardinalities.values())
print("KNN22: min/median/max:", min(vals_k), int(np.median(vals_k)), max(vals_k))

vals_q = list(wq.cardinalities.values())
print("Queen: min/median/max:", min(vals_q), int(np.median(vals_q)), max(vals_q))
print("Queen islands:", vals_q.count(0))

# --- block 16 (raises an error) ---
try:
    import numpy as np, esda
    import warnings
    warnings.filterwarnings('ignore')

    # KNN-22 weights, symmetrized, row-standardized
    wk = KNN.from_dataframe(sel, k=22)
    wk = wk.symmetrize()
    wk.transform = 'R'

    x = sel['AveInteres'].values.astype(float)
    gi = esda.G_Local(x, wk, permutations=999)
    print("Gi* mean z (should ~0):", gi.z.mean())
    print("Gi* z quantiles 0.005/0.025/0.5/0.975/0.995:", np.quantile(gi.z, [0.005,0.025,0.5,0.975,0.995]))
    print("pvals min:", gi.p_sim.min(), "Gi* sig at 95% two-sided:", ((gi.p_sim<0.025)&(gi.p_sim>0)).sum())

    sel['Gi_star'] = gi.stat
    sel['Gi_star_z'] = gi.z
    sel['Gi_star_p'] = gi.p_sim
    sel['Gi_class'] = 'not significant'
    sel.loc[(gi.z >= 1.96), 'Gi_class'] = 'hot spot'
    sel.loc[(gi.z <= -1.96), 'Gi_class'] = 'cold spot'
    sel.loc[(gi.z > 1.6449) & (gi.z < 1.96), 'Gi_class'] = 'high'
    sel.loc[(gi.z < -1.6449) & (gi.z > -1.96), 'Gi_class'] = 'low'
    print(sel['Gi_class'].value_counts())
    gi_res = gi
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 17 ---
print([a for a in dir(gi) if not a.startswith('_')])

# --- block 18 ---
import numpy as np, esda
import warnings
warnings.filterwarnings('ignore')

x = sel['AveInteres'].values.astype(float)
gi = esda.G_Local(x, wk, permutations=999)
print("Gi* mean Zs (should ~0):", gi.Zs.mean())
print("Gi* Zs quantiles 0.005/0.025/0.5/0.975/0.995:", np.quantile(gi.Zs, [0.005,0.025,0.5,0.975,0.995]))
print("Gi* p_sim min:", gi.p_sim.min())

sel['Gi_star'] = gi.Gs
sel['Gi_star_z'] = gi.Zs
sel['Gi_star_p'] = gi.p_sim
sel['Gi_class'] = 'not significant'
sel.loc[(gi.Zs >= 1.96), 'Gi_class'] = 'hot spot (95%)'
sel.loc[(gi.Zs <= -1.96), 'Gi_class'] = 'cold spot (95%)'
sel.loc[(gi.Zs > 1.6449) & (gi.Zs < 1.96), 'Gi_class'] = 'high (90%)'
sel.loc[(gi.Zs < -1.6449) & (gi.Zs > -1.96), 'Gi_class'] = 'low (90%)'
print(sel['Gi_class'].value_counts())
gi_res = gi

# --- block 19 ---
import numpy as np, esda
import warnings
warnings.filterwarnings('ignore')

x = sel['AveInteres'].values.astype(float)

# Queen contiguity, symmetrized, R
wq2 = Queen.from_dataframe(sel)
wq2 = wq2.symmetrize()
wq2.transform = 'R'
gi_q = esda.G_Local(x, wq2, permutations=999)
print("Queen Gi* Zs: min/median/max:", gi_q.Zs.min(), np.median(gi_q.Zs), gi_q.Zs.max())
print("Queen Gi* significant 95% (z>1.96):", (gi_q.Zs>=1.96).sum(), " cold(z<-1.96):", (gi_q.Zs<=-1.96).sum())

# KNN22 for comparison
gi_k = esda.G_Local(x, wk, permutations=999)
print("KNN22 Gi* Zs: min/median/max:", gi_k.Zs.min(), np.median(gi_k.Zs), gi_k.Zs.max())
print("KNN22 Gi* significant 95% (z>1.96):", (gi_k.Zs>=1.96).sum(), " cold(z<-1.96):", (gi_k.Zs<=-1.96).sum())

# --- block 20 (raises an error) ---
try:
    import numpy as np
    from mgwr.gwr import GWR

    # Response: AveInteres (interest rate). Predictors: loan activity variables
    coords = np.column_stack([sel.geometry.centroid.x.values, sel.geometry.centroid.y.values])
    y = sel['AveInteres'].values.astype(float)
    X = sel[['AcceptedLo','pHHLoanRat']].values.astype(float)

    gwr_model = GWR(coords, y, X, bw=22, kernel='bisquare')
    gwr_res = gwr_model.fit()
    print("Global R2:", gwr_res.R2_global)
    print("Local R2 min/max:", gwr_res.R2_local.min(), gwr_res.R2_local.max())
    print("AICc:", gwr_res.AICc)
    print("params shape:", gwr_res.params.shape)
    print(gwr_res.params[:3])
    print("pred range:", gwr_res.predy.min(), gwr_res.predy.max())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 21 ---
import inspect, mgwr.gwr as gwr_mod
print(inspect.getsource(gwr_mod.GWR._local_fit))

# --- block 22 ---
import inspect
from spglm.iwls import _compute_betas_gwr
print(inspect.getsource(_compute_betas_gwr))

# --- block 23 ---
import inspect
from spglm.glm import GLM
print(inspect.getsource(GLM.__init__))

# --- block 24 (raises an error) ---
try:
    import numpy as np
    from mgwr.gwr import GWR
    from spglm.glm import GLM
    from spglm import user

    rng = np.random.default_rng(0)
    coords = rng.normal(size=(50,2))*1000
    y = rng.normal(size=50)
    X = rng.normal(size=(50,2))
    g = GWR(coords, y, X, bw=5, kernel='bisquare')
    print("y shape:", g.y.shape, "y type:", type(g.y))
    print("X shape:", g.X.shape, "X type:", type(g.X))
    print("X[0]:", g.X[0], "shape:", np.asarray(g.X[0]).shape)
    print("constant used:", g.constant)
    wi = g._build_wi(0, 5).reshape(-1,1)
    print("wi shape:", wi.shape)
    from spglm.iwls import _compute_betas_gwr
    betas, invxt = _compute_betas_gwr(g.y, g.X, wi)
    print("betas shape:", betas.shape)
    pred = np.dot(g.X[0], betas)
    print("pred:", pred, type(pred))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 25 ---
import numpy as np
from mgwr.gwr import GWR
from spglm.iwls import _compute_betas_gwr

rng = np.random.default_rng(0)
coords = rng.normal(size=(50,2))*1000
y = rng.normal(size=50)
X = rng.normal(size=(50,2))
g = GWR(coords, y, X, bw=5, kernel='bisquare')
print("y shape:", g.y.shape, "y type:", type(g.y))
print("X shape:", g.X.shape, "X type:", type(g.X))
print("X[0]:", np.asarray(g.X[0]), "shape:", np.asarray(g.X[0]).shape)
wi = g._build_wi(0, 5).reshape(-1,1)
print("wi shape:", wi.shape)
betas, invxt = _compute_betas_gwr(g.y, g.X, wi)
print("betas shape:", betas.shape)
pred = np.dot(g.X[0], betas)
print("pred:", pred, type(pred), np.asarray(pred).shape)

# --- block 26 ---
import numpy as np
a = np.dot(np.array([1.,2.,3.]), np.array([3.,4.,5.]))
print(a, type(a))
try:
    print(a[0])
except Exception as e:
    print("ERROR:", e)
print(np.__version__)

# --- block 27 ---
import numpy as np
from mgwr.gwr import GWR
from spglm.family import Gaussian, Poisson, Binomial
from spglm.iwls import _compute_betas_gwr, iwls

def _local_fit_fixed(self, i):
    wi = self._build_wi(i, self.bw).reshape(-1, 1)
    if isinstance(self.family, Gaussian):
        betas, inv_xtx_xt = _compute_betas_gwr(self.y, self.X, wi)
        predy = float(np.asarray(np.dot(self.X[i], betas)).reshape(-1)[0])
        resid = self.y[i] - predy
        influ = np.asarray(np.dot(self.X[i], inv_xtx_xt[:, i])).reshape(-1)[0]
        w = 1
    elif isinstance(self.family, (Poisson, Binomial)):
        rslt = iwls(self.y, self.X, self.family, self.offset, None,
                    self.fit_params['ini_params'], self.fit_params['tol'],
                    self.fit_params['max_iter'], wi=wi)
        inv_xtx_xt = rslt[5]
        w = rslt[3][i][0]
        influ = rslt[2][i]
        predy = rslt[1][i]
        resid = self.y[i] - predy
        betas = rslt[0]
    if self.fit_params['lite']:
        return influ, resid, predy, betas.reshape(-1)
    else:
        Si = np.dot(self.X[i], inv_xtx_xt).reshape(-1)
        tr_STS_i = np.sum(Si * Si * w * w)
        CCT = np.diag(np.dot(inv_xtx_xt, inv_xtx_xt.T)).reshape(-1)
        if not self.hat_matrix:
            Si = None
        return influ, resid, predy, betas.reshape(-1), w, Si, tr_STS_i, CCT

GWR._local_fit = _local_fit_fixed
print("patched OK")

# --- block 28 (raises an error) ---
try:
    import numpy as np
    from mgwr.gwr import GWR

    coords = np.column_stack([sel.geometry.centroid.x.values, sel.geometry.centroid.y.values])
    y = sel['AveInteres'].values.astype(float)
    X = sel[['AcceptedLo','pHHLoanRat']].values.astype(float)

    gwr_model = GWR(coords, y, X, bw=22, kernel='bisquare', n_jobs=4)
    gwr_res = gwr_model.fit()
    print("Global R2:", round(gwr_res.R2_global, 4))
    print("Local R2: min/median/max:", round(gwr_res.R2_local.min(),4), round(np.median(gwr_res.R2_local),4), round(gwr_res.R2_local.max(),4))
    print("AICc:", round(gwr_res.AICc,1))
    print("params shape:", gwr_res.params.shape)
    print("coef names:", gwr_res.params.columns if hasattr(gwr_res.params,'columns') else None)
    print(gwr_res.params.head(3))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 29 (raises an error) ---
try:
    import numpy as np
    from mgwr.gwr import GWR

    coords = np.column_stack([sel.geometry.centroid.x.values, sel.geometry.centroid.y.values])
    y = sel['AveInteres'].values.astype(float)
    X = sel[['AcceptedLo','pHHLoanRat']].values.astype(float)

    gwr_model = GWR(coords, y, X, bw=22, kernel='bisquare', n_jobs=1)
    gwr_res = gwr_model.fit()
    print("Global R2:", round(gwr_res.R2_global, 4))
    print("Local R2: min/median/max:", round(gwr_res.R2_local.min(),4), round(np.median(gwr_res.R2_local),4), round(gwr_res.R2_local.max(),4))
    print("AICc:", round(gwr_res.AICc,1))
    print("params shape:", gwr_res.params.shape)
    print(gwr_res.params.head(3))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 30 ---
print([a for a in dir(gwr_res) if not a.startswith('_')])

# --- block 31 ---
import numpy as np
print("Global R2:", round(gwr_res.R2,4), " adj R2:", round(gwr_res.adj_R2,4))
print("localR2 min/median/max:", round(gwr_res.localR2.min(),4), round(np.median(gwr_res.localR2),4), round(gwr_res.localR2.max(),4))
print("AICc:", round(gwr_res.aicc,1), " BIC:", round(gwr_res.bic,1))
print("params:\n", gwr_res.params)
print("tvalues:\n", gwr_res.tvalues)
print("pvalues:\n", gwr_res.pvalues)
print("sigma2:", gwr_res.sigma2)
print("predy range:", gwr_res.predy.min(), gwr_res.predy.max())

# --- block 32 ---
import os, warnings
warnings.filterwarnings('ignore')
import geopandas as gpd

os.makedirs('pred_results', exist_ok=True)

# Deliverable 1: selected tracts with >= 30 accepted loans
sel_selected = sel.drop(columns=['Gi_star','Gi_star_z','Gi_star_p','Gi_class'], errors='ignore').copy()
# keep the base loan attributes
sel_selected.to_file('pred_results/selected_loans_30plus.shp', encoding='utf-8')
print("Saved selected:", len(sel_selected), "tracts")
print(sel_selected[['ZIP3','AcceptedLo','AveInteres']].head())

# --- block 33 ---
import numpy as np, warnings
warnings.filterwarnings('ignore')
import geopandas as gpd

# Re-attach Gi* fields (from gi_k = KNN22)
gi = gi_k
sel['Gi_star'] = gi.Gs
sel['Gi_star_z'] = gi.Zs
sel['Gi_star_p'] = gi.p_sim
sel['Gi_class'] = 'not significant'
sel.loc[(gi.Zs >= 1.96), 'Gi_class'] = 'hot spot (95%)'
sel.loc[(gi.Zs <= -1.96), 'Gi_class'] = 'cold spot (95%)'
sel.loc[(gi.Zs > 1.6449) & (gi.Zs < 1.96), 'Gi_class'] = 'high (90%)'
sel.loc[(gi.Zs < -1.6449) & (gi.Zs > -1.96), 'Gi_class'] = 'low (90%)'

print("Gi* classes:"); print(sel['Gi_class'].value_counts())
print("Top hot spots (highest Gi* z):\n", sel.sort_values('Gi_star_z', ascending=False).head(5)[['ZIP3','STATE','AveInteres','Gi_star_z','Gi_class']])

# Deliverable 2: Gi* hot spot layer
sel.to_file('pred_results/interest_hotspots_gi_star.shp', encoding='utf-8')
print("Saved Gi* hot spot layer:", len(sel))

# --- block 34 ---
import numpy as np, esda, warnings
warnings.filterwarnings('ignore')
from libpysal.weights import KNN

x = sel['AveInteres'].values.astype(float)
# binary KNN22
wkb = KNN.from_dataframe(sel, k=22).symmetrize()
wkb.transform = 'B'
gib = esda.G_Local(x, wkb, permutations=999)
print("Binary KNN22 Gi* Zs: min/median/max:", round(gib.Zs.min(),2), round(np.median(gib.Zs),2), round(gib.Zs.max(),2))
print("sig 95% hot:", (gib.Zs>=1.96).sum(), " cold:", (gib.Zs<=-1.96).sum())

# R KNN22 (reference)
print("R KNN22 Gi* Zs max:", round(gi_k.Zs.max(),2))

# --- block 35 (raises an error) ---
try:
    import numpy as np
    from scipy.stats import t as tdist
    import geopandas as gpd
    import warnings
    warnings.filterwarnings('ignore')

    p = gwr_res.params.values          # 808 x 3: const, AcceptedLo, pHHLoanRat
    tv = gwr_res.tvalues.values
    predy = gwr_res.predy.values
    localR2 = gwr_res.localR2.values

    gdf = sel.copy()
    gdf['gwr_const'] = p[:,0]
    gdf['gwr_AcceptedLo'] = p[:,1]
    gdf['gwr_pHHLoanRat'] = p[:,2]
    gdf['gwr_t_AcceptedLo'] = tv[:,1]
    gdf['gwr_t_pHHLoanRat'] = tv[:,2]
    gdf['gwr_p_AcceptedLo'] = 2*(1 - tdist.cdf(np.abs(tv[:,1]), df=19))
    gdf['gdf'] = None
    gdf['gwr_p_pHHLoanRat'] = 2*(1 - tdist.cdf(np.abs(tv[:,2]), df=19))
    gdf['gwr_pred'] = predy
    gdf['gwr_localR2'] = localR2
    gdf['gwr_resid'] = gdf['AveInteres'] - predy

    print("GWR layer saved preview:")
    print(gdf[['ZIP3','gwr_const','gwr_AcceptedLo','gwr_pHHLoanRat','gwr_t_pHHLoanRat','gwr_p_pHHLoanRat','gwr_localR2','gwr_pred']].head())
    print("pHHLoanRat significant (p<0.05) fraction:", (gdf['gwr_p_pHHLoanRat']<0.05).mean())
    print("AcceptedLo significant (p<0.05) fraction:", (gdf['gwr_p_AcceptedLo']<0.05).mean())
    print("localR2 mean:", localR2.mean())
    gdf_gwr = gdf
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 36 ---
import numpy as np
from scipy.stats import t as tdist
import warnings
warnings.filterwarnings('ignore')

p = np.asarray(gwr_res.params)
tv = np.asarray(gwr_res.tvalues)
predy = np.asarray(gwr_res.predy).reshape(-1)
localR2 = np.asarray(gwr_res.localR2).reshape(-1)

gdf = sel.copy()
gdf['gwr_const'] = p[:,0]
gdf['gwr_AcceptedLo'] = p[:,1]
gdf['gwr_pHHLoanRat'] = p[:,2]
gdf['gwr_t_AcceptedLo'] = tv[:,1]
gdf['gwr_t_pHHLoanRat'] = tv[:,2]
gdf['gwr_p_AcceptedLo'] = 2*(1 - tdist.cdf(np.abs(tv[:,1]), df=19))
gdf['gwr_p_pHHLoanRat'] = 2*(1 - tdist.cdf(np.abs(tv[:,2]), df=19))
gdf['gwr_pred'] = predy
gdf['gwr_localR2'] = localR2
gdf['gwr_resid'] = gdf['AveInteres'] - predy

print(gdf[['ZIP3','gwr_const','gwr_AcceptedLo','gwr_pHHLoanRat','gwr_t_pHHLoanRat','gwr_p_pHHLoanRat','gwr_localR2','gwr_pred']].head())
print("pHHLoanRat sig(p<.05) frac:", round((gdf['gwr_p_pHHLoanRat']<0.05).mean(),3))
print("AcceptedLo sig(p<.05) frac:", round((gdf['gwr_p_AcceptedLo']<0.05).mean(),3))
print("localR2 mean:", round(localR2.mean(),3))
gdf_gwr = gdf

# --- block 37 (raises an error) ---
try:
    import numpy as np
    import geopandas as gpd
    import warnings
    warnings.filterwarnings('ignore')

    gwr_layer = gdf_gwr.copy()
    # shapefile-safe names
    gwr_layer = gwr_layer.rename(columns={
        'gwr_const':'gwr_const',
        'gwr_AcceptedLo':'gwr_AccLo',
        'gwr_pHHLoanRat':'gwr_HHLoanR',
        'gwr_t_AcceptedLo':'gwr_t_AccLo',
        'gwr_t_pHHLoanRat':'gwr_t_HHLoanR',
        'gwr_p_AcceptedLo':'gwr_p_AccLo',
        'gwr_p_pHHLoanRat':'gwr_p_HHLoanR',
        'gwr_pred':'gwr_pred',
        'gwr_localR2':'gwr_localR2',
        'gwr_resid':'gwr_resid',
    })
    # keep relevant fields for the shapefile
    keep = ['ZIP3','STATE','AcceptedLo','AveInteres','pHHLoanRat','AveLoanGra',
            'Gi_star','Gi_star_z','Gi_star_p','Gi_class',
            'gwr_const','gwr_AccLo','gwr_HHLoanR','gwr_t_AccLo','gwr_t_HHLoanR',
            'gwr_p_AccLo','gwr_p_HHLoanR','gwr_pred','gwr_localR2','gwr_resid']
    gwr_layer[keep].to_file('pred_results/gwr_results.shp', encoding='utf-8')
    print("Saved GWR layer with columns:")
    print(list(gwr_layer[keep].columns))
    print("rows:", len(gwr_layer))
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 38 ---
import warnings
warnings.filterwarnings('ignore')
cols = ['ZIP3','STATE','AcceptedLo','AveInteres','pHHLoanRat','AveLoanGra',
        'Gi_star','Gi_star_z','Gi_star_p','Gi_class',
        'gwr_const','gwr_AccLo','gwr_HHLoanR','gwr_t_AccLo','gwr_t_HHLoanR',
        'gwr_p_AccLo','gwr_p_HHLoanR','gwr_pred','gwr_localR2','gwr_resid','geometry']
gwr_layer[cols].to_file('pred_results/gwr_results.shp', encoding='utf-8')
print("Saved GWR layer, rows:", len(gwr_layer))
print(list(gwr_layer[cols].drop(columns=['geometry']).columns))

# --- block 39 ---
import numpy as np

p = np.asarray(gwr_res.params); tv = np.asarray(gwr_res.tvalues)
sig_hh = (np.abs(tv[:,2]) > 1.96).mean()
sig_acc = (np.abs(tv[:,1]) > 1.96).mean()

txt = f"""GEOGRAPHICALLY WEIGHTED REGRESSION (GWR)
Response (key field): AveInteres (average interest rate, %)
Predictors: AcceptedLo (accepted loans, count), pHHLoanRat (household-loan ratio)
Data: tracts with AcceptedLo >= 30  (n = {len(sel)})
Bandwidth/neighborhood: adaptive, k = 22 nearest neighbors
Kernel: bisquare; estimation: iterative weighted least squares (IWLS)
Global regression R2 = {gwr_res.R2:.4f}  (adjusted R2 = {gwr_res.adj_R2:.4f})
Local R2: mean = {np.mean(gwr_res.localR2):.4f}, median = {np.median(gwr_res.localR2):.4f},
          min = {np.min(gwr_res.localR2):.4f}, max = {np.max(gwr_res.localR2):.4f}
AICc = {gwr_res.aicc:.1f} ; BIC = {gwr_res.bic:.1f} ; sigma^2 = {gwr_res.sigma2:.4f}

LOCAL COEFFICIENTS (statistics over the {len(sel)} calibration locations)
Constant:        mean = {p[:,0].mean():.4f}, sd = {p[:,0].std():.4f}
AcceptedLo:      mean = {p[:,1].mean():.2e}, sd = {p[:,1].std():.2e}
                 significant (|t|>1.96) at {100*sig_acc:.1f}% of locations
pHHLoanRat:      mean = {p[:,2].mean():.4f}, sd = {p[:,2].std():.4f}
                 significant (|t|>1.96) at {100*sig_hh:.1f}% of locations

INTERPRETATION
- Location matters for interest rates: the GWR local R2 (mean {100*np.mean(gwr_res.localR2):.1f}%)
  shows the model fits each tract's interest rate well using nearby tracts only, i.e.
  interest-rate relationships are locally structured.
- pHHLoanRat (share of households with loans) has a consistently POSITIVE local effect:
  mean local coefficient = {p[:,2].mean():+.4f} with a positive sign at
  {(p[:,2] > 0).mean()*100:.1f}% of locations and statistically significant at {100*sig_hh:.1f}%.
  Areas where more households hold loans experience higher average interest rates.
- AcceptedLo (volume of accepted loans) shows a slightly negative local coefficient
  (mean {p[:,1].mean():+.2e}) that is statistically significant at only {100*sig_acc:.1f}% of
  locations, so loan volume is a weak local driver of interest rates.
- The spatially varying coefficients indicate that the interest-rate surface is
  driven primarily by local credit-demand density (household-loan ratio) rather
  than by the raw number of accepted loans.

HOT SPOT (Getis-Ord Gi*) ANALYSIS
Weights: 22 nearest neighbors (symmetrized, row-standardized), 999 permutations
Max Gi* z = {gi_k.Zs.max():.3f}  ->  no tracts exceed the 95% critical value (1.96);
no statistically significant hot or cold spots of AveInteres were detected.
This indicates that average interest rates are spatially uniform at the scale of a
22-tract neighborhood, consistent with a small spatial scale of variation.
"""
with open('pred_results/gwr_model_summary.txt','w') as f:
    f.write(txt)
print(txt)

# --- block 40 (raises an error) ---
try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import warnings
    warnings.filterwarnings('ignore')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.suptitle('Interest rates (AveInteres) on tracts with >= 30 accepted loans  (n=808)', fontsize=14)

    # Panel 1: Gi* hot spot z
    ax = axes[0]
    gdf_gwr.plot(column='Gi_star_z', ax=ax, cmap='RdYlGn_r', legend=True,
                 legend_kwds={'label': 'Gi* z-score'}, missing_kwds={'color':'lightgray'})
    ax.set_title('Hot spots: Getis-Ord Gi* (KNN-22)', fontsize=11)
    ax.set_axis_off()

    # Panel 2: local GWR coefficient of pHHLoanRat
    ax = axes[1]
    gdf_gwr.plot(column='gwr_HHLoanR', ax=ax, cmap='coolwarm', legend=True,
                 legend_kwds={'label': 'GWR coeff. pHHLoanRat'})
    ax.set_title('GWR local coefficient: pHHLoanRat (k=22)', fontsize=11)
    ax.set_axis_off()

    # Panel 3: local R2
    ax = axes[2]
    gdf_gwr.plot(column='gwr_localR2', ax=ax, cmap='YlGn', legend=True,
                 legend_kwds={'label': 'GWR local R$^2$'}, missing_kwds={'color':'lightgray'})
    ax.set_title('GWR local R$^2$ (k=22)', fontsize=11)
    ax.set_axis_off()

    plt.tight_layout()
    plt.savefig('pred_results/gwr_hotspots_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("figure saved")
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 41 ---
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
print(list(gdf_gwr.columns))

# --- block 42 ---
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
fig.suptitle('Interest rates (AveInteres) on tracts with >= 30 accepted loans  (n=808)', fontsize=14)

ax = axes[0]
gdf_gwr.plot(column='Gi_star_z', ax=ax, cmap='RdYlGn_r', legend=True,
             legend_kwds={'label': 'Gi* z-score'}, missing_kwds={'color':'lightgray'})
ax.set_title('Hot spots: Getis-Ord Gi* (KNN-22)', fontsize=11)
ax.set_axis_off()

ax = axes[1]
gdf_gwr.plot(column='gwr_pHHLoanRat', ax=ax, cmap='coolwarm', legend=True,
             legend_kwds={'label': 'GWR coeff. pHHLoanRat'})
ax.set_title('GWR local coefficient: pHHLoanRat (k=22)', fontsize=11)
ax.set_axis_off()

ax = axes[2]
gdf_gwr.plot(column='gwr_localR2', ax=ax, cmap='YlGn', legend=True,
             legend_kwds={'label': 'GWR local R$^2$'}, missing_kwds={'color':'lightgray'})
ax.set_title('GWR local R$^2$ (k=22)', fontsize=11)
ax.set_axis_off()

plt.tight_layout()
plt.savefig('pred_results/gwr_hotspots_map.png', dpi=150, bbox_inches='tight')
plt.close()
print("figure saved")

# --- block 43 (raises an error) ---
try:
    import geopandas as gpd
    import warnings
    warnings.filterwarnings('ignore')

    s = gpd.read_file('pred_results/selected_loans_30plus.shp')
    g = gpd.read_file('pred_results/gwr_results.shp')
    h = gpd.read_file('pred_results/interest_hotspots_gi_star.shp')
    print("selected:", s.shape, "| AcceptedLo min/max:", s['AcceptedLo'].min(), s['AcceptedLo'].max())
    print("gwr:", g.shape)
    print("  gwr_pred range:", g['gwr_pred'].min(), g['gwr_pred'].max())
    print("  localR2 mean:", g['gwr_localR2'].mean())
    print("  pHHLoanRat coeff mean:", g['gwr_HHLoanR'].mean())
    print("hotspots:", h.shape, "| Gi_star_z range:", h['Gi_star_z'].min(), h['Gi_star_z'].max())
    print("Gi* classes:", h['Gi_class'].value_counts().to_dict())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 44 ---
print(list(g.columns))

# --- block 45 (raises an error) ---
try:
    import warnings
    warnings.filterwarnings('ignore')

    # Use the in-memory gdf_gwr (full names) and rename to <=10 chars
    gl = gdf_gwr.rename(columns={
        'gwr_AcceptedLo':'gwr_AccLo',
        'gwr_pHHLoanRat':'gwr_HHlr',
        'gwr_t_AcceptedLo':'gwr_t_AccLo',
        'gwr_t_pHHLoanRat':'gwr_t_HHlr',
        'gwr_p_AcceptedLo':'gwr_p_AccLo',
        'gwr_p_pHHLoanRat':'gwr_p_HHlr',
        'gwr_localR2':'gwr_lr',
    })
    cols = ['ZIP3','STATE','AcceptedLo','AveInteres','pHHLoanRat','AveLoanGra',
            'Gi_star','Gi_star_z','Gi_star_p','Gi_class',
            'gwr_const','gwr_AccLo','gwr_HHlr','gwr_t_AccLo','gwr_t_HHlr',
            'gwr_p_AccLo','gwr_p_HHlr','gwr_pred','gwr_lr','gwr_resid','geometry']
    assert all(len(c) <= 10 for c in cols[:-1])
    gl[cols].to_file('pred_results/gwr_results.shp', encoding='utf-8')

    # verify
    import geopandas as gpd
    gg = gpd.read_file('pred_results/gwr_results.shp')
    print(list(gg.columns))
    print("gwr_pred range:", gg['gwr_pred'].min(), gg['gwr_pred'].max())
    print("gwr_lr mean:", gg['gwr_lr'].mean())
    print("gwr_HHlr mean:", gg['gwr_HHlr'].mean())
    h = gpd.read_file('pred_results/interest_hotspots_gi_star.shp')
    print("hotspots Gi_star_z range:", h['Gi_star_z'].min(), h['Gi_star_z'].max(), "| classes:", h['Gi_class'].value_counts().to_dict())
except BaseException as _e:
    print('block failed:', repr(_e)[:200])

# --- block 46 ---
import warnings
warnings.filterwarnings('ignore')
import geopandas as gpd

gl = gdf_gwr.rename(columns={
    'gwr_AcceptedLo':'gwr_AccLo',
    'gwr_pHHLoanRat':'gwr_HHlr',
    'gwr_t_AcceptedLo':'gwr_tA',
    'gwr_t_pHHLoanRat':'gwr_tH',
    'gwr_p_AcceptedLo':'gwr_pA',
    'gwr_p_pHHLoanRat':'gwr_pH',
    'gwr_localR2':'gwr_lr',
})
cols = ['ZIP3','STATE','AcceptedLo','AveInteres','pHHLoanRat','AveLoanGra',
        'Gi_star','Gi_star_z','Gi_star_p','Gi_class',
        'gwr_const','gwr_AccLo','gwr_HHlr','gwr_tA','gwr_tH',
        'gwr_pA','gwr_pH','gwr_pred','gwr_lr','gwr_resid','geometry']
print("max len:", max(len(c) for c in cols[:-1]))
gl[cols].to_file('pred_results/gwr_results.shp', encoding='utf-8')

gg = gpd.read_file('pred_results/gwr_results.shp')
print(list(gg.columns))
print("gwr_pred range:", round(gg['gwr_pred'].min(),3), round(gg['gwr_pred'].max(),3))
print("gwr_lr mean:", round(gg['gwr_lr'].mean(),4))
print("gwr_HHlr mean:", round(gg['gwr_HHlr'].mean(),4))
h = gpd.read_file('pred_results/interest_hotspots_gi_star.shp')
print("hotspots Gi_star_z range:", round(h['Gi_star_z'].min(),3), round(h['Gi_star_z'].max(),3))
print("classes:", h['Gi_class'].value_counts().to_dict())
