# Correct: ruggedness and road distance both rescaled by a 10-class quantile reverse reclassification
# (not MSLarge); land cover weighted 1.25.
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


import rasterio, numpy as np
ps = rasterio.open('dataset/Protected_Status.tif').read(1)
el_ds = rasterio.open('dataset/Elevation.tif'); el = el_ds.read(1).astype(float)
print(el_ds.nodata, el_ds.transform, lc_meta['transform'], rasterio.open('dataset/Protected_Status.tif').transform)
print(np.unique(lc, return_counts=True)); print(np.unique(ps, return_counts=True))
print(np.nanmin(el), np.nanmax(el))

print(open('dataset/Elevation.tif.aux.xml').read()[:2400])
v,c=np.unique(el,return_counts=True); print(v[:5],c[:5],v[-5:],c[-5:])

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


from rasterio.features import rasterize
from scipy import ndimage
tr = lc_meta['transform']; shp = lc.shape
road_r = rasterize(((g,1) for g in roads.geometry), out_shape=shp, transform=tr, fill=0, all_touched=True, dtype='uint8')
road_dist = ndimage.distance_transform_edt(road_r==0)*30.0
rug = ndimage.maximum_filter(el,size=3,mode='nearest')-ndimage.minimum_filter(el,size=3,mode='nearest')
print(road_r.sum(), np.percentile(road_dist,[0,25,50,75,90,100]))
print(np.percentile(rug,[0,10,25,50,75,90,99,100]))

nod = (lc==0)|(lc==255)
r,c_=np.where(nod); print(r.min(),r.max(),c_.min(),c_.max(), (el[nod]==0).mean())
print(hab[['core','Name']])

def eq_int(a, n=10, reverse=False):
    mn, mx = a.min(), a.max()
    cls = np.clip(np.floor((a-mn)/(mx-mn)*n)+1,1,n)
    return (n+1-cls) if reverse else cls
rug_cost = eq_int(rug, reverse=True)            # more rugged -> lower cost
road_cost = eq_int(road_dist, reverse=True)     # closer to road -> higher cost
lc_map = {11:10,21:6,22:7,23:8,24:9,31:5,41:2,42:1,43:2,52:1,71:3,81:5,82:6,90:4,95:4}
lc_cost = np.full(shp, np.nan)
for k,v_ in lc_map.items(): lc_cost[lc==k]=v_
ps_map = {1:1,2:3,3:6,4:9,0:10,255:10}
ps_cost = np.zeros(shp)
for k,v_ in ps_map.items(): ps_cost[ps==k]=v_
cost = 1.25*rug_cost + 1.0*road_cost + 1.25*lc_cost + 1.0*ps_cost
for n_,a in [('rug',rug_cost),('road',road_cost),('lc',lc_cost),('ps',ps_cost),('cost',cost)]:
    print(n_, np.unique(a[~np.isnan(a)],return_counts=True) if n_!='cost' else (np.nanmin(a),np.nanmean(a),np.nanmax(a)))

def quant(a, n=10, reverse=False):
    valid = ~np.isnan(cost)
    br = np.unique(np.percentile(a[valid], np.linspace(0,100,n+1)[1:-1]))
    cls = np.digitize(a, br, right=True)+1
    cls = 1 + (cls-1)*(n-1)/max(cls.max()-1,1)  # stretch to 1..n
    return (n+1-cls) if reverse else cls
rug_cost = np.round(quant(rug, reverse=True))
road_cost = np.round(quant(road_dist, reverse=True))
for a in (rug_cost, road_cost): print(np.unique(a, return_counts=True))
cost = 1.25*rug_cost + 1.0*road_cost + 1.25*lc_cost + 1.0*ps_cost
print(np.nanmin(cost), np.nanmean(cost), np.nanmax(cost))

from skimage.graph import MCP_Geometric
hab_r = rasterize(((g,i+1) for i,g in enumerate(hab.geometry)), out_shape=shp, transform=tr, fill=0, all_touched=True, dtype='uint8')
cst = np.where(np.isnan(cost), np.inf, cost)
H = len(hab); pair = {}; paths = {}
for i in range(H):
    src = list(zip(*np.where(hab_r==i+1)))
    m = MCP_Geometric(cst, fully_connected=True)
    acc, _ = m.find_costs(src)
    for j in range(H):
        if j==i: continue
        cells = np.argwhere(hab_r==j+1)
        vals = acc[cells[:,0],cells[:,1]]; k_ = np.argmin(vals)
        if (j,i) not in pair or vals[k_] < pair[(j,i)]:
            pair[(i,j)] = vals[k_]; paths[(i,j)] = m.traceback(tuple(cells[k_]))
    print(i, hab.Name[i], len(src))
for k_,v_ in pair.items(): print(k_, hab.Name[k_[0]], '->', hab.Name[k_[1]], round(v_))

import networkx as nx, geopandas as gpd, matplotlib.pyplot as plt, os
from shapely.geometry import LineString
G = nx.Graph()
for (i,j),w in pair.items(): G.add_edge(i,j,weight=w)
mst = nx.minimum_spanning_tree(G)
rows=[]
for i,j in mst.edges():
    p = paths[(i,j)] if (i,j) in paths else paths[(j,i)]
    xy = [tr*(c+0.5,r+0.5) for r,c in p]
    rows.append({'from':hab.Name[i],'to':hab.Name[j],'cost':pair.get((i,j),pair.get((j,i))),'geometry':LineString(xy)})
conn = gpd.GeoDataFrame(rows, crs=hab.crs); conn['length_km']=conn.length/1000
print(conn.drop(columns='geometry'))
os.makedirs('pred_results',exist_ok=True)
conn.to_file('pred_results/optimal_connections.geojson', driver='GeoJSON')
ext = [tr.c, tr.c+tr.a*shp[1], tr.f+tr.e*shp[0], tr.f]
fig,ax = plt.subplots(figsize=(13,10))
im = ax.imshow(cost, extent=ext, cmap='RdYlGn_r'); plt.colorbar(im, ax=ax, shrink=0.7, label='Weighted cost (low = favourable)')
hab.plot(ax=ax, facecolor='none', edgecolor='blue', linewidth=2)
conn.plot(ax=ax, color='black', linewidth=2.5)
for _,r_ in hab.iterrows():
    pt=r_.geometry.representative_point(); ax.annotate(r_.Name,(pt.x,pt.y),fontsize=11,fontweight='bold',color='navy',xytext=(5,5),textcoords='offset points')
ax.set_title('Optimal region connections between mountain lion habitats\n(cost = 1.25·ruggedness + 1.0·road distance + 1.25·land cover + 1.0·protected status)')
ax.set_xlabel('Easting (m, UTM 11N)'); ax.set_ylabel('Northing (m)')
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([],[],color='black',lw=2.5,label='Optimal corridor'),Line2D([],[],color='blue',lw=2,label='Habitat core')],loc='upper right')
plt.savefig('pred_results/optimal_path.png', dpi=150, bbox_inches='tight'); plt.close()
with rasterio.open('pred_results/cost_surface.tif','w',driver='GTiff',height=shp[0],width=shp[1],count=1,dtype='float32',crs=hab.crs,transform=tr,nodata=-9999) as d:
    d.write(np.where(np.isnan(cost),-9999,cost).astype('float32'),1)
print(os.listdir('pred_results'))