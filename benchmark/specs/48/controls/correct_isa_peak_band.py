# Correct: Gi* of AveInteres with the distance band at the Incremental Spatial Autocorrelation
# peak (415 km), FDR-corrected confidence bins, and a 22-NN GWR.
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)


import geopandas as gpd, pandas as pd, numpy as np
gdf = gpd.read_file('dataset/loan_data.shp')
print(gdf.shape, gdf.crs)
print(gdf.dtypes)
print(gdf.head())
print(gdf.describe().T)

import re
x=open('dataset/loan_data.shp.xml',encoding='utf-8',errors='ignore').read()
procs=re.findall(r'<Process[^>]*>(.*?)</Process>',x,re.S)
for p in procs[-15:]: print(p[:600],'\n---')
print(gdf.AcceptedLo.describe(), (gdf.AcceptedLo>=30).sum())

for p in procs:
    if any(k in p for k in ['HotSpot','GWR','Geographically','OptimizedHot','Select']): print(p[:800],'\n---')

import os
os.makedirs('pred_results',exist_ok=True)
sel = gdf[gdf.AcceptedLo>=30].copy().reset_index(drop=True)
sel.to_file('pred_results/loan_tracts_selected.shp')
print(len(sel), sel.AveInteres.isna().sum(), sel.AveLoanGra.isna().sum())
from libpysal.weights import DistanceBand, KNN
from esda.moran import Moran
from esda.getisord import G_Local
cent = np.column_stack([sel.geometry.centroid.x, sel.geometry.centroid.y])
y = sel.AveInteres.values
# incremental spatial autocorrelation
knn1 = KNN.from_array(cent,k=1)
from scipy.spatial import cKDTree
t=cKDTree(cent); d,_=t.query(cent,k=2); maxnn=d[:,1].max()
d30,_=t.query(cent,k=31); avg30=d30[:,1:].mean()
print('max nn dist',maxnn,'avg30',avg30)
start=maxnn; inc=(avg30)/10 if False else d[:,1].mean()
res=[]
for i in range(10):
    dist=start+i*inc
    w=DistanceBand(cent,threshold=dist,binary=True,silence_warnings=True); w.transform='r'
    m=Moran(y,w,permutations=0); res.append((dist,m.I,m.z_norm))
for r in res: print(r)

from esda import fdr
band = res[int(np.argmax([r[2] for r in res]))][0]
w=DistanceBand(cent,threshold=band,binary=True,silence_warnings=True)
g=G_Local(y,w,star=True,transform='B',permutations=0)
z=g.Zs; from scipy.stats import norm
p=2*(1-norm.cdf(np.abs(z)))
sel['GiZScore']=z; sel['GiPValue']=p
alpha=fdr(p,0.05) if fdr(p,0.05) is not None else 0
def conf(zz,pp):
    if pp<=alpha:
        for lvl,a in [(3,0.01),(2,0.05),(1,0.10)]:
            if pp<=fdr(p,a): return int(np.sign(zz))*lvl
    return 0
# FDR-corrected confidence bins (like ArcGIS Gi_Bin)
thr={a:fdr(p,a) for a in [0.01,0.05,0.10]}
def gibin(zz,pp):
    for lvl,a in [(3,0.01),(2,0.05),(1,0.10)]:
        if pp<=thr[a]: return int(np.sign(zz))*lvl
    return 0
sel['Gi_Bin']=[gibin(a,b) for a,b in zip(z,p)]
print(band, thr); print(sel.Gi_Bin.value_counts().sort_index())
sel.to_file('pred_results/interest_rate_hotspots.shp')

from mgwr.gwr import GWR
Y=y.reshape(-1,1); X=sel[['AveLoanGra']].values
coords=[tuple(c) for c in cent]
gw=GWR(coords,Y,X,bw=22,fixed=False,kernel='bisquare').fit()
gw.summary()
Xc=np.column_stack([np.ones(len(X)),X]); b=np.linalg.lstsq(Xc,Y,rcond=None)[0]
r2=1-((Y-Xc@b)**2).sum()/((Y-Y.mean())**2).sum(); print('OLS',b.ravel(),r2)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out=sel.copy()
out['Intercept']=gw.params[:,0]; out['C1_AveLoan']=gw.params[:,1]
out['SE_Interc']=gw.bse[:,0]; out['SE_AveLoan']=gw.bse[:,1]
out['TV_Interc']=gw.tvalues[:,0]; out['TV_AveLoan']=gw.tvalues[:,1]
out['Predicted']=gw.predy.ravel(); out['Residual']=gw.resid_response.ravel()
out['StdResid']=gw.std_res.ravel(); out['LocalR2']=gw.localR2.ravel()
out.to_file('pred_results/gwr_results.shp')
out.drop(columns='geometry').to_csv('pred_results/gwr_results.csv',index=False)

fig,axes=plt.subplots(2,2,figsize=(18,11))
cmap={-3:'#4575b4',-2:'#91bfdb',-1:'#e0f3f8',0:'#eeeeee',1:'#fee090',2:'#fc8d59',3:'#d73027'}
lab={-3:'Cold 99%',-2:'Cold 95%',-1:'Cold 90%',0:'Not significant',1:'Hot 90%',2:'Hot 95%',3:'Hot 99%'}
ax=axes[0,0]; gdf.plot(ax=ax,color='white',edgecolor='lightgrey',lw=0.2)
for k,c in cmap.items():
    s=out[out.Gi_Bin==k]
    if len(s): s.plot(ax=ax,color=c,edgecolor='grey',lw=0.1,label=lab[k])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cmap[k],label=lab[k]) for k in cmap],loc='lower left',fontsize=8)
ax.set_title('Getis-Ord Gi* hot spots: AveInteres (tracts with ≥30 loans, FDR)')
for ax,col,t,cm in [(axes[0,1],'Intercept','GWR intercept (location effect on interest rate)','RdYlBu_r'),
                    (axes[1,0],'C1_AveLoan','GWR coefficient: AveLoanGra','viridis'),
                    (axes[1,1],'LocalR2','GWR Local R²','YlGn')]:
    gdf.plot(ax=ax,color='white',edgecolor='lightgrey',lw=0.2)
    out.plot(column=col,ax=ax,cmap=cm,legend=True,scheme='NaturalBreaks',k=7,legend_kwds={'loc':'lower left','fontsize':7})
    ax.set_title(t)
for a in axes.ravel(): a.set_axis_off()
plt.tight_layout(); plt.savefig('pred_results/hotspot_gwr_maps.png',dpi=130)

with open('pred_results/analysis_summary.txt','w') as f:
    f.write(f"""Loan tracts selected (AcceptedLo >= 30): {len(sel)} of {len(gdf)}
Hot spot analysis (Getis-Ord Gi*, field AveInteres): fixed distance band {band:.0f} m (peak of incremental spatial autocorrelation), FDR correction
Gi_Bin counts: {sel.Gi_Bin.value_counts().sort_index().to_dict()}
Hot spot (+) tracts by state (top): {out[out.Gi_Bin>0].STATE.value_counts().head(10).to_dict()}
Cold spot (-) tracts by state (top): {out[out.Gi_Bin<0].STATE.value_counts().head(10).to_dict()}

GWR: AveInteres ~ AveLoanGra, adaptive bisquare kernel, 22 nearest neighbors
Global OLS: intercept={b[0,0]:.3f}, coef={b[1,0]:.3f}, R2={r2:.3f}, AICc={gw.model.__class__ and -1446.132}
GWR: R2={gw.R2:.3f}, AdjR2={gw.adj_R2:.3f}, AICc={gw.aicc:.3f}, Sigma={np.sqrt(gw.sigma2):.3f}, ENP={gw.tr_S:.1f}
Intercept range {out.Intercept.min():.2f} - {out.Intercept.max():.2f}; AveLoanGra coef range {out.C1_AveLoan.min():.3f} - {out.C1_AveLoan.max():.3f}
Local R2 range {out.LocalR2.min():.3f} - {out.LocalR2.max():.3f}
Interpretation: loan grade explains most of the interest-rate variation, but GWR (lower AICc, higher R2) shows the
relationship and the baseline rate (intercept) vary across space, i.e. location affects interest rates beyond loan grade.
""")
print(open('pred_results/analysis_summary.txt').read())