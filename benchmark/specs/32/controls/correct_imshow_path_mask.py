# Correct: the corridors are drawn as a raster, ax.imshow(np.where(path_mask, 1, nan)) - a drawn image
# whose sparse cells are the path raster.
import os, rasterio, numpy as np, pandas as pd, geopandas as gpd
os.makedirs('pred_results', exist_ok=True)
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
prot = _src.read().astype(float) if _src.count > 1 else _src.read(1).astype(float)
prot_meta = dict(crs=_src.crs, transform=_src.transform, width=_src.width, height=_src.height,
               nodata=_src.nodata, dtype=_src.dtypes[0], count=_src.count, bounds=_src.bounds, res=_src.res)
if _src.nodata is not None:
    prot[prot == _src.nodata] = np.nan
print(f"Size: {_src.width} x {_src.height}, bands: {_src.count}, dtype: {_src.dtypes[0]}")
print(f"CRS: {_src.crs.to_string() if _src.crs else None}, resolution: {_src.res}")
print(f"Bounds: {tuple(round(float(_b), 4) for _b in _src.bounds)}")
print(f"NoData: {_src.nodata} (replaced by NaN in prot)")
_valid = prot[~np.isnan(prot)]
print(f"Valid values: {_valid.size} of {prot.size}" + (f", min={_valid.min():.4g}, max={_valid.max():.4g}, mean={_valid.mean():.4g}" if _valid.size else ""))
print("Variables: prot = float array" + (" (bands, rows, cols)" if _src.count > 1 else " (rows, cols)") + ", prot_meta = dict(crs, transform, nodata, ...)")
_src.close()


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



import geopandas as gpd, pandas as pd, numpy as np
lcv = gpd.read_file('dataset/landCover.tif.vat.dbf') if False else None
from pyogrio import read_dataframe
lcv = read_dataframe('dataset/landCover.tif.vat.dbf')
pv = read_dataframe('dataset/Protected_Status.tif.vat.dbf')
print(lcv.to_string()); print(pv.to_string())



print(np.unique(elev, return_counts=True)[0][:20], len(np.unique(elev)))
print(np.percentile(elev,[0,1,5,25,50,75,95,99,100]))
u,c=np.unique(prot,return_counts=True); print(dict(zip(u,c)))
u,c=np.unique(lc,return_counts=True); print(dict(zip(u.astype(int),c)))
print(elev_meta['transform'], elev_meta.get('nodata'))
print(hab[['OBJECTID','core','Name']])



from rasterio import features
from rasterio.transform import rowcol
m0 = (lc==0)|(lc==255)
rows,cols=np.where(m0)
print('lc 0/255 rows',rows.min(),rows.max(),'cols',cols.min(),cols.max(), 'frac on edge', np.mean((rows==0)|(rows==lc.shape[0]-1)|(cols==0)|(cols==lc.shape[1]-1)))
m1=(prot==0)|(prot==255)
r1,c1=np.where(m1); print('prot 0/255 frac on edge', np.mean((r1==0)|(r1==prot.shape[0]-1)|(c1==0)|(c1==prot.shape[1]-1)), 'overlap with lc0/255', np.mean(m0[m1]))
print(hab.crs, hab.total_bounds)



from scipy import ndimage
from scipy.ndimage import distance_transform_edt
H,W=elev.shape
# 1. Ruggedness = Focal std (3x3) of elevation
k=np.ones((3,3))/9
mu=ndimage.convolve(elev,k,mode='nearest')
mu2=ndimage.convolve(elev**2,k,mode='nearest')
rug=np.sqrt(np.maximum(mu2-mu**2,0))
lo,hi=np.percentile(rug,[1,99])
rug_n=np.clip((rug-lo)/(hi-lo),0,1)
cost_elev = 1 + 9*(1-rug_n)          # higher ruggedness -> lower cost
print('rug range',rug.min(),rug.max(),'pcts',lo,hi)
# 2. Road distance
road_r = features.rasterize(((g,1) for g in roads.geometry if g is not None), out_shape=(H,W), transform=elev_meta['transform'], fill=0, all_touched=True, dtype='uint8')
print('road cells',road_r.sum())
dist_road = distance_transform_edt(road_r==0, sampling=30)   # metres
print('dist range',dist_road.min(),dist_road.max(), np.percentile(dist_road,[25,50,75]))
cost_road = 1 + 9*np.clip(1 - dist_road/2000.0, 0, 1)  # closer to road -> higher cost, 0 beyond 2 km



lc_map = {11:10, 21:5, 22:7, 23:9, 24:10, 31:4, 41:1, 42:1, 43:1, 52:2, 71:2, 81:6, 82:7, 90:3, 95:3, 0:10, 255:10}
lci = lc.astype(int)
cost_lc = np.full((H,W), np.nan)
for v,c in lc_map.items(): cost_lc[lci==v]=c
print('unmapped', np.isnan(cost_lc).sum())
prot_map = {1:1, 2:3, 3:6, 4:9, 0:9, 255:9}
pi = prot.astype(int)
cost_prot = np.full((H,W), np.nan)
for v,c in prot_map.items(): cost_prot[pi==v]=c
print('unmapped prot', np.isnan(cost_prot).sum())
w = {'lc':0.40,'elev':0.20,'road':0.25,'prot':0.15}
assert abs(sum(w.values())-1)<1e-9
final_cost = w['lc']*cost_lc + w['elev']*cost_elev + w['road']*cost_road + w['prot']*cost_prot
print('final cost range', final_cost.min(), final_cost.max(), np.percentile(final_cost,[5,50,95]))



from skimage.graph import MCP_Geometric
import time
hab_r = features.rasterize([(g,int(i)) for g,i in zip(hab.geometry, hab.OBJECTID)], out_shape=(H,W), transform=elev_meta['transform'], fill=0, dtype='int32')
ids = [int(i) for i in hab.OBJECTID]
print({i:(hab_r==i).sum() for i in ids})
names = dict(zip(hab.OBJECTID, hab.Name))
t=time.time()
cum={}; mcps={}
for i in ids:
    starts=[tuple(p) for p in np.argwhere(hab_r==i)]
    mcp=MCP_Geometric(final_cost, fully_connected=True)
    cc,_=mcp.find_costs(starts)
    cum[i]=cc; mcps[i]=mcp
print('time',time.time()-t)
# region-to-region minimum cost
pair={}
for i in ids:
    for j in ids:
        if i<j:
            pair[(i,j)] = cum[i][hab_r==j].min()
for k,v in pair.items(): print(names[k[0]],names[k[1]], round(v,1))



import networkx as nx
G=nx.Graph()
for (i,j),v in pair.items(): G.add_edge(i,j,weight=v)
T=nx.minimum_spanning_tree(G, weight='weight')
paths=[]
for i,j in T.edges():
    a,b = (i,j) if cum[i][hab_r==j].min() <= cum[j][hab_r==i].min() else (j,i)
    # source region a, target region b
    cells=np.argwhere(hab_r==b); vals=cum[a][cells[:,0],cells[:,1]]
    end=tuple(cells[np.argmin(vals)])
    p=np.array(mcps[a].traceback(end))
    paths.append((a,b,p,cum[a][end]))
    print(names[a],'->',names[b],'cost',round(cum[a][end],1),'length cells',len(p))
path_mask=np.zeros((H,W),bool)
for a,b,p,_ in paths: path_mask[p[:,0],p[:,1]]=True
print('path cells', path_mask.sum())



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio, os, shapely
from shapely.geometry import LineString
os.makedirs('pred_results',exist_ok=True)
tr=elev_meta['transform']
ext=[tr.c, tr.c+tr.a*W, tr.f+tr.e*H, tr.f]
fig,axes=plt.subplots(1,2,figsize=(16,9))
ax=axes[0]
im=ax.imshow(final_cost,cmap='viridis_r',extent=ext)
plt.colorbar(im,ax=ax,fraction=0.035,label='Final cost (1=low, 10=high)')
hab.boundary.plot(ax=ax,color='red',linewidth=1)
for _,r in hab.iterrows(): ax.annotate(r.Name,(r.geometry.centroid.x,r.geometry.centroid.y),color='white',fontsize=9,ha='center')
ax.set_title('Weighted final cost surface + habitats'); ax.set_xlabel('Easting (m)'); ax.set_ylabel('Northing (m)')
ax=axes[1]
ax.imshow(np.where(path_mask,1.0,np.nan),cmap='autumn',extent=ext,interpolation='nearest')
ax.imshow(np.where(path_mask,np.nan,1.0),cmap='Greys',alpha=0.25,extent=ext)
hab.plot(ax=ax,column='Name',cmap='tab10',edgecolor='k',alpha=0.6,legend=True,legend_kwds={'loc':'lower left','fontsize':8})
for a,b,p,c in paths:
    pass
for _,r in hab.iterrows(): ax.annotate(r.Name,(r.geometry.centroid.x,r.geometry.centroid.y),color='black',fontsize=9,ha='center',weight='bold')
ax.set_title('Optimal region connections (minimum spanning network)'); ax.set_xlabel('Easting (m)')
plt.tight_layout(); plt.savefig('pred_results/optimal_path.png',dpi=150); plt.close()
# outputs
meta=dict(elev_meta); meta.update(dtype='float32',count=1,nodata=None)
with rasterio.open('pred_results/final_cost.tif','w',**meta) as dst: dst.write(final_cost.astype('float32'),1)
with rasterio.open('pred_results/optimal_path_raster.tif','w',**{**meta,'dtype':'uint8'}) as dst: dst.write(path_mask.astype('uint8'),1)
lines=[]
for a,b,p,c in paths:
    xy=[rasterio.transform.xy(tr,r,cc) for r,cc in p]
    lines.append(dict(from_region=names[a],to_region=names[b],cost=float(c),geometry=LineString(xy)))
gpd.GeoDataFrame(lines,crs='EPSG:26911').to_file('pred_results/optimal_paths.geojson',driver='GeoJSON')
print(os.listdir('pred_results'))
