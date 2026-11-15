# Correct: Gi* confidence-bin map drawn one class at a time (one single-colour PatchCollection per bin).

import geopandas as gpd
p=gpd.read_file('dataset/High_Blood_Level_Results.shp');z=gpd.read_file('dataset/Sacramento_ZIP_Codes.shp')
print(p.crs,z.crs,len(p),p.total_bounds,z.total_bounds,p.geom_type.unique())



import numpy as np, pandas as pd, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from libpysal.weights import DistanceBand
from esda.getisord import G_Local
from esda.moran import Moran
from scipy.stats import norm
c='EPSG:32610'
P=p.to_crs(c);Z=z.to_crs(c)
# hex grid
s=800 # side length m
minx,miny,maxx,maxy=Z.total_bounds
w=np.sqrt(3)*s;h=1.5*s
hexes=[]
r=0
y=miny-s
while y<maxy+s:
    x=minx-w+(w/2 if r%2 else 0)
    while x<maxx+w:
        hexes.append(Polygon([(x+s*np.sin(np.radians(a)),y+s*np.cos(np.radians(a))) for a in range(0,360,60)]))
        x+=w
    y+=h;r+=1
H=gpd.GeoDataFrame(geometry=hexes,crs=c)
H=H[H.intersects(Z.union_all())].reset_index(drop=True)
j=gpd.sjoin(P,H,predicate='within')
H['count']=j.groupby('index_right').size().reindex(H.index).fillna(0)
print(len(H),H['count'].sum(),H['count'].describe())
cen=np.c_[H.centroid.x,H.centroid.y]
# optimal distance: peak z of Moran's I incremental
y_=H['count'].values
res=[]
for d in np.arange(w*1.05,8000,w/2):
    try:
        wt=DistanceBand(cen,threshold=d,binary=True,silent=True)
        if wt.islands: continue
        m=Moran(y_,wt,permutations=0);res.append((d,m.z_norm))
    except Exception as e: pass
print(res)



from scipy.spatial import cKDTree
nn=cKDTree(cen).query(cen,k=2)[0][:,1].max();print(nn)
res=[]


res=[]
for d in np.arange(nn*1.01,nn+9000,500):
    wt=DistanceBand(cen,threshold=d,binary=True)
    m=Moran(y_,wt,permutations=0);res.append((d,m.z_norm,len(wt.islands)))
for r_ in res:print(r_)



d=2900
wt=DistanceBand(cen,threshold=d,binary=True)
g=G_Local(y_,wt,transform='B',star=True,permutations=999,seed=1)
H['z']=g.Zs;H['p']=2*(1-norm.cdf(abs(g.Zs)))
def cl(zv):
    a=abs(zv)
    n=3 if a>=2.58 else 2 if a>=1.96 else 1 if a>=1.645 else 0
    return n*np.sign(zv)
H['gi_bin']=[int(cl(v)) for v in H.z]
print(H.gi_bin.value_counts())
cols={3:'#b2182b',2:'#ef8a62',1:'#fddbc7',0:'#f0f0f0',-1:'#d1e5f0',-2:'#67a9cf',-3:'#2166ac'}
lab={3:'Hot spot 99%',2:'Hot spot 95%',1:'Hot spot 90%',0:'Not significant',-1:'Cold spot 90%',-2:'Cold spot 95%',-3:'Cold spot 99%'}
fig,ax=plt.subplots(figsize=(9,11))
Zw=Z.to_crs(c)
for k in sorted(cols,reverse=True):
    s_=H[H.gi_bin==k]
    if len(s_):s_.plot(ax=ax,color=cols[k],edgecolor='white',linewidth=.4,label=f'{lab[k]} ({len(s_)})')
Zw.boundary.plot(ax=ax,color='gray',linewidth=.7)
H[H['count']>0].apply(lambda r:ax.annotate(int(r['count']),(r.geometry.centroid.x,r.geometry.centroid.y),ha='center',va='center',fontsize=5,color='k'),axis=1)
ax.legend(title='Gi* hot spot (hexagon bins)',loc='lower left',fontsize=8)
ax.set_title(f'Elevated Blood Lead Levels in Children – Sacramento\nHexagon-aggregated Hot Spot Analysis (Gi*, distance band {d} m; n=373 cases)')
ax.set_axis_off();plt.tight_layout()
import os;os.makedirs('pred_results',exist_ok=True)
plt.savefig('pred_results/blood_lead_hotspots.png',dpi=200)
H.to_file('pred_results/blood_lead_hex_hotspots.gpkg',driver='GPKG')



from matplotlib.patches import Patch
hs=[Patch(facecolor=cols[k],edgecolor='gray',label=f'{lab[k]} ({(H.gi_bin==k).sum()})') for k in sorted(cols,reverse=True) if (H.gi_bin==k).any()]
ax.legend(handles=hs,title='Gi* hexagon classes\n(numbers = cases per hex)',loc='lower left',fontsize=8)
fig.savefig('pred_results/blood_lead_hotspots.png',dpi=200)
