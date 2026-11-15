# Correct: linearly rescaled ruggedness (5x5 focal std) and a banded road-distance cost. The best-fitting
# road variant of the cost check misses the ruggedness weight narrowly; the MSLarge road variant meets
# every threshold.
import os, rasterio, numpy as np, pandas as pd, geopandas as gpd
os.makedirs('pred_results', exist_ok=True)

import rasterio, geopandas as gpd, numpy as np
for f in ['landCover','Protected_Status','Elevation']:
    r=rasterio.open(f'dataset/{f}.tif'); a=r.read(1); print(f,r.crs,r.res,r.bounds,r.nodata,a.dtype,np.unique(a)[:40])
h=gpd.read_file('dataset/habitats.geojson'); print(h, h.crs)
ro=gpd.read_file('dataset/Roads.geojson'); print(ro.shape, ro.crs, ro.columns.tolist())



from scipy import ndimage as ndi
from rasterio import features
from skimage.graph import MCP_Geometric
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
os.makedirs('pred_results',exist_ok=True)
lc=rasterio.open('dataset/landCover.tif'); T=lc.transform; shp=lc.shape
L=lc.read(1); P=rasterio.open('dataset/Protected_Status.tif').read(1); E=rasterio.open('dataset/Elevation.tif').read(1).astype(float)
print(np.unique(E)[-5:], (E==255).sum(), (L==255).sum(), (L==0).sum())
# ruggedness: 5x5 focal std
m=ndi.uniform_filter(E,5); rug=np.sqrt(np.maximum(ndi.uniform_filter(E**2,5)-m**2,0))
def scale(x,lo,hi): return lo+(x-x.min())/(x.max()-x.min())*(hi-lo)
rug_c=scale(rug,10,1)   # high ruggedness -> low cost
# roads distance
rr=features.rasterize(((g,1) for g in ro.geometry),out_shape=shp,transform=T,all_touched=True,dtype='uint8')
d=ndi.distance_transform_edt(rr==0,sampling=30)
print(d.max())
road_c=np.interp(d,[0,90,300,1000,2000,1e9],[10,9,7,4,1,1])
lcmap={11:10,21:8,22:9,23:10,24:10,31:3,41:1,42:1,43:1,52:2,71:2,81:6,82:7,90:1,95:3}
lc_c=np.full(shp,10.0)
for k,v in lcmap.items(): lc_c[L==k]=v
pm={1:1,2:2,3:4,4:6}
p_c=np.full(shp,8.0)
for k,v in pm.items(): p_c[P==k]=v
cost=0.35*lc_c+0.25*road_c+0.2*rug_c+0.2*p_c
invalid=(L==0)|(L==255)
cost[invalid]=np.nan
print(np.nanmin(cost),np.nanmax(cost))



Ev=E.copy(); bad=E==255
idx=ndi.distance_transform_edt(bad,return_distances=False,return_indices=True)
Ef=E[tuple(idx)]
m=ndi.uniform_filter(Ef,5); rug=np.sqrt(np.maximum(ndi.uniform_filter(Ef**2,5)-m**2,0))
rug_c=scale(rug,10,1)
cost=0.35*lc_c+0.25*road_c+0.2*rug_c+0.2*p_c
hm=features.rasterize(((g,i+1) for i,g in enumerate(h.geometry)),out_shape=shp,transform=T,dtype='uint8')
print([(i,(hm==i).sum(),(invalid&(hm==i)).sum()) for i in range(1,5)])
c=cost.copy(); c[invalid&(hm==0)]=np.inf
# pairwise least cost paths
import itertools,networkx as nx
res={}
for i in range(1,5):
    mcp=MCP_Geometric(c,fully_connected=True)
    src=np.argwhere(hm==i)
    cum,_=mcp.find_costs(src)
    for j in range(i+1,5):
        tg=np.argwhere(hm==j); k=np.argmin(cum[tuple(tg.T)]); end=tg[k]
        path=np.array(mcp.traceback(tuple(end)))
        res[(i,j)]=(cum[tuple(end)],path)
        print(i,j,cum[tuple(end)])
G=nx.Graph(); [G.add_edge(i,j,weight=v[0]) for (i,j),v in res.items()]
mst=nx.minimum_spanning_tree(G); print(mst.edges(data=True))



ext=[T.c,T.c+T.a*shp[1],T.f+T.e*shp[0],T.f]
fig,ax=plt.subplots(figsize=(11,9))
cm=np.ma.masked_where(invalid,cost)
im=ax.imshow(cm,extent=ext,cmap='YlGnBu_r' if False else 'viridis_r',alpha=.9)
plt.colorbar(im,label='Weighted cost (low = good)',shrink=.7)
ro.plot(ax=ax,color='grey',linewidth=.2,alpha=.5)
h.plot(ax=ax,color='orange',edgecolor='k')
for (i,j) in mst.edges:
    p=res[(i,j)][1] if (i,j) in res else res[(j,i)][1]
    ax.plot(T.c+(p[:,1]+.5)*T.a,T.f+(p[:,0]+.5)*T.e,'r',lw=2,label='Optimal path' if (i,j)==list(mst.edges)[0] else None)
for k,g in enumerate(h.geometry): ax.text(g.centroid.x,g.centroid.y,str(k+1),fontsize=12,weight='bold')
ax.set_xlim(ext[0],ext[1]);ax.set_ylim(ext[2],ext[3]);ax.legend();ax.set_title('Optimal connections between mountain lion habitats')
plt.savefig('pred_results/optimal_path.png',dpi=150,bbox_inches='tight')
