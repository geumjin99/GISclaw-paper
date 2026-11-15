# Correct: ruggedness and road distance rescaled by decile classes (not MSLarge), road cells set to the
# maximum cost.
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


_src = rasterio.open('dataset/Elevation.tif')
elev = _src.read().astype(float) if _src.count > 1 else _src.read(1).astype(float)
elev_meta = dict(crs=_src.crs, transform=_src.transform, width=_src.width, height=_src.height,
               nodata=_src.nodata, dtype=_src.dtypes[0], count=_src.count, bounds=_src.bounds, res=_src.res)
if _src.nodata is not None:
    elev[elev == _src.nodata] = np.nan
print(f"Size: {_src.width} x {_src.height}, bands: {_src.count}, dtype: {_src.dtypes[0]}")
print(f"CRS: {_src.crs.to_string() if _src.crs else None}, resolution: {_src.res}")
print(f"Bounds: {tuple(round(float(_b), 4) for _b in _src.bounds)}")
print(f"NoData: {_src.nodata} (replaced by NaN in elev)")
_valid = elev[~np.isnan(elev)]
print(f"Valid values: {_valid.size} of {elev.size}" + (f", min={_valid.min():.4g}, max={_valid.max():.4g}, mean={_valid.mean():.4g}" if _valid.size else ""))
print("Variables: elev = float array" + (" (bands, rows, cols)" if _src.count > 1 else " (rows, cols)") + ", elev_meta = dict(crs, transform, nodata, ...)")
_src.close()


_src = rasterio.open('dataset/landCover.tif')
lc = _src.read().astype(float) if _src.count > 1 else _src.read(1).astype(float)
lc_meta = dict(crs=_src.crs, transform=_src.transform, width=_src.width, height=_src.height,
               nodata=_src.nodata, dtype=_src.dtypes[0], count=_src.count, bounds=_src.bounds, res=_src.res)
if _src.nodata is not None:
    lc[lc == _src.nodata] = np.nan
print(f"Size: {_src.width} x {_src.height}, bands: {_src.count}, dtype: {_src.dtypes[0]}")
print(f"CRS: {_src.crs.to_string() if _src.crs else None}, resolution: {_src.res}")
print(f"Bounds: {tuple(round(float(_b), 4) for _b in _src.bounds)}")
print(f"NoData: {_src.nodata} (replaced by NaN in lc)")
_valid = lc[~np.isnan(lc)]
print(f"Valid values: {_valid.size} of {lc.size}" + (f", min={_valid.min():.4g}, max={_valid.max():.4g}, mean={_valid.mean():.4g}" if _valid.size else ""))
print("Variables: lc = float array" + (" (bands, rows, cols)" if _src.count > 1 else " (rows, cols)") + ", lc_meta = dict(crs, transform, nodata, ...)")
_src.close()


_src = rasterio.open('dataset/Protected_Status.tif')
ps = _src.read().astype(float) if _src.count > 1 else _src.read(1).astype(float)
ps_meta = dict(crs=_src.crs, transform=_src.transform, width=_src.width, height=_src.height,
               nodata=_src.nodata, dtype=_src.dtypes[0], count=_src.count, bounds=_src.bounds, res=_src.res)
if _src.nodata is not None:
    ps[ps == _src.nodata] = np.nan
print(f"Size: {_src.width} x {_src.height}, bands: {_src.count}, dtype: {_src.dtypes[0]}")
print(f"CRS: {_src.crs.to_string() if _src.crs else None}, resolution: {_src.res}")
print(f"Bounds: {tuple(round(float(_b), 4) for _b in _src.bounds)}")
print(f"NoData: {_src.nodata} (replaced by NaN in ps)")
_valid = ps[~np.isnan(ps)]
print(f"Valid values: {_valid.size} of {ps.size}" + (f", min={_valid.min():.4g}, max={_valid.max():.4g}, mean={_valid.mean():.4g}" if _valid.size else ""))
print("Variables: ps = float array" + (" (bands, rows, cols)" if _src.count > 1 else " (rows, cols)") + ", ps_meta = dict(crs, transform, nodata, ...)")
_src.close()


import numpy as np
for n,a in [('elev',elev),('lc',lc),('ps',ps)]:
    u,c=np.unique(a,return_counts=True); print(n, len(u), list(zip(u[:40].astype(int),c[:40])) if len(u)<40 else (u[:10],u[-10:]))
print(open('dataset/Elevation.tif.aux.xml').read()[:2500])

hab = gpd.read_file('dataset/habitats.geojson')
print(f"Rows: {len(hab)}")
print(f"CRS: {hab.crs.to_string() if hab.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in hab.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in hab.total_bounds]}")
_cols = [c for c in hab.columns if c != hab.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(hab[_cols].head(3).to_string())


roads = gpd.read_file('dataset/Roads.geojson')
print(f"Rows: {len(roads)}")
print(f"CRS: {roads.crs.to_string() if roads.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in roads.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in roads.total_bounds]}")
_cols = [c for c in roads.columns if c != roads.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(roads[_cols].head(3).to_string())


print(hab[['OBJECTID','core','Name','Shape_Area']])
print(roads.Carto_Desc.value_counts())

from scipy import ndimage
from rasterio import features
T=elev_meta['transform']; shape=elev.shape
# 1 Ruggedness: focal range 3x3
rng = ndimage.maximum_filter(elev,3)-ndimage.minimum_filter(elev,3)
rug_cost = 10 - 9*(rng-rng.min())/(rng.max()-rng.min())   # high ruggedness -> low cost
# 2 Distance to roads
rr = features.rasterize(((g,1) for g in roads.geometry), out_shape=shape, transform=T, fill=0, all_touched=True, dtype='uint8')
dist = ndimage.distance_transform_edt(rr==0)*30.0
road_cost = 1 + 9*(dist.max()-dist)/(dist.max())             # near roads -> high cost
# 3 Land cover
lc_map={11:10,21:8,22:9,23:10,24:10,31:7,41:2,42:1,43:2,52:1,71:4,81:6,82:7,90:3,95:5,0:10,255:10}
lc_cost=np.vectorize(lambda v: lc_map.get(int(v),10))(lc) if False else np.full(shape,10.0)
for k,v in lc_map.items(): lc_cost[lc==k]=v
# 4 Protected
ps_cost=np.full(shape,10.0)
for k,v in {1:1,2:3,3:6,4:9}.items(): ps_cost[ps==k]=v
for n,a in [('rng',rng),('rug',rug_cost),('dist',dist),('road',road_cost),('lc',lc_cost),('ps',ps_cost)]:
    print(n, np.nanmin(a), np.nanmax(a), np.nanmean(a).round(2))

def qclass(a):
    q=np.quantile(a,np.linspace(0,1,11)[1:-1]); return np.digitize(a,np.unique(q))+1  # 1..10
rq=qclass(rng); dq=qclass(dist)
print(np.unique(rq,return_counts=True)); print(np.unique(dq,return_counts=True))
rug_cost = (11 - rq).astype(float)                 # most rugged -> 1
road_cost = (11 - dq).astype(float)                # farthest -> 1
road_cost[rr==1]=10
W={'rug':1.25,'road':1.0,'lc':1.25,'ps':1.0}
cost = W['rug']*rug_cost + W['road']*road_cost + W['lc']*lc_cost + W['ps']*ps_cost
print(cost.min(), cost.max(), cost.mean())

from scipy.stats import rankdata
def pclass(a):
    r=rankdata(a.ravel(),method='average')/a.size
    return np.clip(np.ceil(r*10),1,10).reshape(a.shape)
rq=pclass(rng); dq=pclass(dist)
print(np.unique(rq,return_counts=True)); print(np.unique(dq,return_counts=True))
rug_cost = 11 - rq; road_cost = 11 - dq
cost = 1.25*rug_cost + 1.0*road_cost + 1.25*lc_cost + 1.0*ps_cost
print(cost.min(), cost.max(), cost.mean())

from skimage.graph import MCP_Geometric
import itertools, time
hab_r = features.rasterize(((g,i+1) for i,g in enumerate(hab.geometry)), out_shape=shape, transform=T, fill=0, dtype='uint8')
print(np.unique(hab_r,return_counts=True))
res={}
for i in range(len(hab)):
    t=time.time()
    m=MCP_Geometric(cost, fully_connected=True)
    starts=np.argwhere(hab_r==i+1)
    acc,_=m.find_costs([tuple(s) for s in starts])
    for j in range(len(hab)):
        if j<=i: continue
        cells=np.argwhere(hab_r==j+1); k=np.argmin(acc[hab_r==j+1]); end=tuple(cells[k])
        path=np.array(m.traceback(end))
        res[(i,j)]=(acc[end]*30, path)
        print(hab.Name[i],'->',hab.Name[j], round(acc[end]*30), len(path))
    print(time.time()-t)

import networkx as nx, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from shapely.geometry import LineString
G=nx.Graph()
for (i,j),(c,p) in res.items(): G.add_edge(i,j,weight=c)
mst=nx.minimum_spanning_tree(G)
rows=[]
for i,j in mst.edges():
    key=(min(i,j),max(i,j)); c,p=res[key]
    xs,ys=rasterio.transform.xy(T,p[:,0],p[:,1])
    rows.append({'from':hab.Name[key[0]],'to':hab.Name[key[1]],'acc_cost':round(c,1),'geometry':LineString(zip(xs,ys))})
paths=gpd.GeoDataFrame(rows,crs=hab.crs); paths['length_m']=paths.length.round(0)
print(paths.drop(columns='geometry'))
import os; os.makedirs('pred_results',exist_ok=True)
paths.to_file('pred_results/optimal_paths.geojson',driver='GeoJSON')
prof=dict(driver='GTiff',height=shape[0],width=shape[1],count=1,dtype='float32',crs=elev_meta['crs'],transform=T)
with rasterio.open('pred_results/final_cost.tif','w',**prof) as d: d.write(cost.astype('float32'),1)
ext=[T.c,T.c+T.a*shape[1],T.f+T.e*shape[0],T.f]
fig,ax=plt.subplots(figsize=(13,10))
im=ax.imshow(cost,extent=ext,cmap='RdYlGn_r'); plt.colorbar(im,ax=ax,shrink=0.7,label='Weighted cost (ruggedness 1.25, roads 1, land cover 1.25, protection 1)')
hab.plot(ax=ax,facecolor='royalblue',edgecolor='navy',alpha=0.8)
for _,r in hab.iterrows():
    pt=r.geometry.representative_point(); ax.annotate(r.Name,(pt.x,pt.y),xytext=(5,5),textcoords='offset points',fontsize=11,weight='bold',color='navy',bbox=dict(fc='white',alpha=0.7))
paths.plot(ax=ax,color='black',linewidth=2.5,label='Optimal corridor')
ax.legend(loc='lower left'); ax.set_title('Optimal Region Connections between Mountain Lion Habitats'); ax.set_xlabel('Easting (m), UTM 11N'); ax.set_ylabel('Northing (m)')
plt.savefig('pred_results/optimal_path.png',dpi=150,bbox_inches='tight'); plt.close()
print(os.listdir('pred_results'))