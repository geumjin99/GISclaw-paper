# Correct: road-distance cost from manual distance bins (30 m ... 3 km) and ruggedness cost from quantile
# classes (not MSLarge).
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



import numpy as np, rasterio, geopandas as gpd
with rasterio.open('dataset/Elevation.tif') as s: elev=s.read(1); meta=s.meta.copy(); tr=s.transform
with rasterio.open('dataset/landCover.tif') as s: lc=s.read(1)
with rasterio.open('dataset/Protected_Status.tif') as s: ps=s.read(1)
print(meta)
for n,a in [('elev',elev),('lc',lc),('ps',ps)]:
    u,c=np.unique(a,return_counts=True); print(n, a.shape, len(u), list(zip(u,c))[:40])
hab=gpd.read_file('dataset/habitats.geojson'); print(hab[['core','Name']], hab.crs)



from rasterio.features import rasterize
from scipy import ndimage as ndi
roads=gpd.read_file('dataset/Roads.geojson')
print(len(roads), roads.crs, roads.Carto_Desc.value_counts())
rr=rasterize(((g,1) for g in roads.geometry if g is not None), out_shape=elev.shape, transform=tr, fill=0, all_touched=True, dtype='uint8')
dist=ndi.distance_transform_edt(rr==0)*30.0
print(rr.sum(), dist.min(), dist.max(), np.percentile(dist,[25,50,75,90,99]))



# Roads: closer = higher cost
bins=[30,60,120,250,500,1000,1500,2000,3000]   # m
road_cost=(10-np.digitize(dist,bins,right=False)).astype(float)
# Ruggedness: 3x3 focal range of elevation; higher ruggedness = lower cost
e=elev.astype(float)
rug=ndi.maximum_filter(e,3,mode='nearest')-ndi.minimum_filter(e,3,mode='nearest')
print('rug pct',np.percentile(rug,[0,10,50,90,99,100]))
edges=np.linspace(rug.min(),rug.max(),11)[1:-1]   # equal interval, 10 classes
rug_cost=(10-np.digitize(rug,edges)).astype(float)
# Land cover (NLCD) reclass
lc_map={11:10,21:8,22:9,23:10,24:10,31:6,41:2,42:1,43:2,52:1,71:4,81:6,82:7,90:3,95:4,0:10,255:10}
lc_cost=np.vectorize(lc_map.get)(lc).astype(float)
# Protected status reclass (0/255 = no status -> highest cost)
ps_map={1:1,2:3,3:6,4:9,0:10,255:10}
ps_cost=np.vectorize(ps_map.get)(ps).astype(float)
for n,a in [('road',road_cost),('rug',rug_cost),('lc',lc_cost),('ps',ps_cost)]:
    u,c=np.unique(a,return_counts=True); print(n,dict(zip(u.astype(int),c)))



qe=np.unique(np.percentile(rug,np.arange(10,100,10)))
print(qe)
rug_cls=np.digitize(rug,qe,right=True)       # 0..len(qe)
rug_cost=10-np.round(rug_cls*9/rug_cls.max())   # high ruggedness -> 1, flat -> 10
u,c=np.unique(rug_cost,return_counts=True); print(dict(zip(u.astype(int),c)))
w={'lc':1.25,'rug':1.25,'road':1.0,'ps':1.0}
cost=w['lc']*lc_cost+w['rug']*rug_cost+w['road']*road_cost+w['ps']*ps_cost
print(cost.min(),cost.max(),cost.mean())
m=meta.copy(); m.update(dtype='float32')
import os; os.makedirs('pred_results',exist_ok=True)
with rasterio.open('pred_results/cost_surface.tif','w',**m) as d: d.write(cost.astype('float32'),1)



from skimage.graph import MCP_Geometric
import networkx as nx
hab_r=rasterize(((g,int(c)) for g,c in zip(hab.geometry,hab.core)),out_shape=elev.shape,transform=tr,fill=0,dtype='uint8')
print({k:int((hab_r==k).sum()) for k in hab.core})
cores=sorted(hab.core.tolist()); acc={}; mcps={}
for k in cores:
    mcp=MCP_Geometric(cost,fully_connected=True)
    starts=np.argwhere(hab_r==k)
    a,_=mcp.find_costs(starts); acc[k]=a*30.0; mcps[k]=mcp
G=nx.Graph(); paths={}
for i in cores:
    for j in cores:
        if j<=i: continue
        cells=np.argwhere(hab_r==j); vals=acc[i][hab_r==j]
        end=tuple(cells[np.argmin(vals)]); G.add_edge(i,j,weight=float(vals.min()))
        paths[(i,j)]=np.array(mcps[i].traceback(end))
mst=nx.minimum_spanning_tree(G)
names=dict(zip(hab.core,hab.Name))
for i,j,d in G.edges(data=True): print(names[i],'-',names[j],round(d['weight']), 'MST' if mst.has_edge(i,j) else '')



from shapely.geometry import LineString
import matplotlib.pyplot as plt
recs=[]
for (i,j),p in paths.items():
    xs=tr.c+(p[:,1]+0.5)*30; ys=tr.f-(p[:,0]+0.5)*30
    recs.append(dict(from_core=i,to_core=j,from_name=names[i],to_name=names[j],acc_cost=G[i][j]['weight'],
                     optimal=mst.has_edge(i,j),geometry=LineString(zip(xs,ys))))
lines=gpd.GeoDataFrame(recs,crs=hab.crs); lines['length_m']=lines.length
opt=lines[lines.optimal]; print(opt[['from_name','to_name','acc_cost','length_m']])
opt.to_file('pred_results/optimal_paths.geojson',driver='GeoJSON')
fig,ax=plt.subplots(figsize=(12,10))
ext=[tr.c,tr.c+30*cost.shape[1],tr.f-30*cost.shape[0],tr.f]
im=ax.imshow(cost,extent=ext,cmap='YlOrRd',alpha=0.85)
plt.colorbar(im,ax=ax,shrink=0.7,label='Weighted cost (low = favourable)')
hab.plot(ax=ax,facecolor='green',edgecolor='darkgreen',alpha=0.7)
lines[~lines.optimal].plot(ax=ax,color='grey',linestyle='--',linewidth=1,label='Other least-cost paths')
opt.plot(ax=ax,color='blue',linewidth=2.5,label='Optimal region connections')
for _,r in hab.iterrows():
    c=r.geometry.centroid; ax.annotate(r.Name,(c.x,c.y),fontsize=11,fontweight='bold',ha='center')
ax.legend(loc='lower left'); ax.set_title('Optimal connections between mountain lion habitats')
ax.set_xlabel('Easting (m, UTM 11N)'); ax.set_ylabel('Northing (m)')
plt.savefig('pred_results/optimal_path.png',dpi=150,bbox_inches='tight'); plt.close()
print(os.listdir('pred_results'))
