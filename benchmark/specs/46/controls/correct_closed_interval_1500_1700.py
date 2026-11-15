# Correct: peak window read as the closed interval [15:00, 17:00] (hours 15-16 plus
# CRASH_TM == "17:00"); SWM file parsed by hand, Gi* on the network weights with the row
# maximum as self weight.
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
cr = gpd.read_file('dataset/crashes.shp')
rd = gpd.read_file('dataset/roads.shp')
print(cr.shape, rd.shape, cr.crs, rd.crs)
print(cr.columns.tolist())
print(rd.columns.tolist())
print(cr.head(3).T)



print(cr.CRASH_HOD.value_counts().sort_index().to_dict())
print(cr.CRASH_DOW.value_counts().to_dict())
print(rd.head())
print(rd.LengthMete.describe())



with open('dataset/nwswm360ft.swm','rb') as f: head=f.read(400)
print(head[:300])



import struct
data=open('dataset/nwswm360ft.swm','rb').read()
nl=data.index(b'\n'); print(data[:nl].decode())
pos=nl+1
n,rowstd=struct.unpack('<ii',data[pos:pos+8]); pos+=8
print(n,rowstd)
neigh={}; wts={}
for _ in range(n):
    mid,k=struct.unpack('<ii',data[pos:pos+8]); pos+=8
    if k>0:
        nb=np.frombuffer(data,'<i4',k,pos); pos+=4*k
        w=np.frombuffer(data,'<f8',k,pos); pos+=8*k
        s=struct.unpack('<d',data[pos:pos+8])[0]; pos+=8
    else: nb=np.array([],int); w=np.array([])
    neigh[mid]=nb.tolist(); wts[mid]=w.tolist()
print(pos,len(data))
ks=np.array([len(v) for v in neigh.values()]); print(pd.Series(ks).describe())
print(rd.UniqID.min(), rd.UniqID.max(), rd.UniqID.nunique(), set(neigh)==set(rd.UniqID))
print(list(neigh.items())[:3], list(wts.items())[:3])



import os; os.makedirs('pred_results',exist_ok=True)
wk=~cr.CRASH_DOW.isin(['Saturday','Sunday'])
peak=cr[wk & (cr.CRASH_HOD.isin([15,16]) | (cr.CRASH_TM=='17:00'))].copy()
print(len(peak))
peak.to_file('pred_results/peak_crashes.shp')
# snap within 0.25 mile = 1320 ft
j=gpd.sjoin_nearest(peak[['CrashID','geometry']], rd[['UniqID','geometry']], max_distance=1320, how='left', distance_col='snapdist')
j=j[~j.index.duplicated()]
print(j.UniqID.isna().sum(), 'not snapped')
from shapely.ops import nearest_points
rgeom=rd.set_index('UniqID').geometry
snapped=[]
for g,u in zip(j.geometry,j.UniqID):
    if pd.isna(u): snapped.append(g)
    else:
        L=rgeom[int(u)]; snapped.append(L.interpolate(L.project(g)))
peak_snap=peak.copy(); peak_snap['geometry']=snapped; peak_snap['UniqID']=j.UniqID.values
peak_snap.to_file('pred_results/peak_crashes_snapped.shp')
cnt=j.dropna(subset=['UniqID']).groupby('UniqID').size()
rj=rd.copy(); rj['Join_Count']=rj.UniqID.map(cnt).fillna(0).astype(int)
rj['LengthMile']=rj.LengthMete/1609.344
rj['CrashRate']=rj.Join_Count/rj.LengthMile
print(rj.Join_Count.sum(), rj.Join_Count.describe(), rj.CrashRate.describe())



from scipy import sparse
from scipy.stats import norm
ids=rj.UniqID.values; idx={u:i for i,u in enumerate(ids)}
rows=[];cols=[];vals=[]
for m,nb in neigh.items():
    i=idx[m]; w=wts[m]
    sw=max(w) if len(w) else 1.0
    rows.append(i);cols.append(i);vals.append(sw)
    for b,ww in zip(nb,w):
        rows.append(i);cols.append(idx[b]);vals.append(ww)
n=len(ids)
W=sparse.csr_matrix((vals,(rows,cols)),shape=(n,n))
x=rj.CrashRate.values.astype(float)
xb=x.mean(); S=np.sqrt((x**2).mean()-xb**2)
Wi=np.asarray(W.sum(1)).ravel(); W2=np.asarray(W.multiply(W).sum(1)).ravel()
z=(W@x - xb*Wi)/(S*np.sqrt((n*W2-Wi**2)/(n-1))); p=2*norm.sf(np.abs(z))
def gbin(z,p):
    b=np.zeros(len(z),int)
    for lvl,a in [(1,0.10),(2,0.05),(3,0.01)]:
        b[(p<=a)&(z>0)]=lvl; b[(p<=a)&(z<0)]=-lvl
    return b
rj['GiZScore']=z; rj['GiPValue']=p; rj['Gi_Bin']=gbin(z,p)
print(rj.Gi_Bin.value_counts().sort_index())



import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
rj.to_file('pred_results/crash_hotspots.shp')
rj.drop(columns='geometry').to_csv('pred_results/crash_hotspots.csv',index=False)
colors={-3:'#4575b5',-2:'#849eba',-1:'#c0ccbe',0:'#d9d9d9',1:'#fab984',2:'#ed7551',3:'#d62f27'}
labels={3:'Hot Spot - 99% Confidence',2:'Hot Spot - 95% Confidence',1:'Hot Spot - 90% Confidence',0:'Not Significant',-1:'Cold Spot - 90%',-2:'Cold Spot - 95%',-3:'Cold Spot - 99%'}
fig,ax=plt.subplots(figsize=(9,14))
rj[rj.Gi_Bin==0].plot(ax=ax,color=colors[0],linewidth=0.3)
for b in [1,2,3]:
    s=rj[rj.Gi_Bin==b]
    if len(s): s.plot(ax=ax,color=colors[b],linewidth=1.6)
present=sorted(rj.Gi_Bin.unique(),reverse=True)
ax.legend(handles=[Patch(color=colors[b],label=f"{labels[b]} ({(rj.Gi_Bin==b).sum()})") for b in present],loc='lower left',title='Gi* Bin (Crash rate per mile)')
ax.set_title('Peak-hour (weekday 3–5 PM) crash hot spots\nBrevard County, FL, 2010–2015 (Getis-Ord Gi*, network weights)')
ax.set_axis_off(); plt.tight_layout(); plt.savefig('pred_results/crash_hotspot_map.png',dpi=200); plt.close()
print(rj.sort_values('GiZScore',ascending=False)[['UniqID','Join_Count','CrashRate','GiZScore','Gi_Bin']].head(10))
print(os.listdir('pred_results'))
