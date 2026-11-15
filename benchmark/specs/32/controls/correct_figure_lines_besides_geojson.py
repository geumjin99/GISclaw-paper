# Correct: the MST corridors are written to a GeoJSON and also drawn with ax.plot in optimal_path.png,
# so the map shows the corridors as figure lines.
import os, rasterio, numpy as np, pandas as pd, geopandas as gpd
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



import geopandas as gpd, numpy as np, pandas as pd
lc_vat = gpd.read_file('dataset/landCover.tif.vat.dbf')
prot_vat = gpd.read_file('dataset/Protected_Status.tif.vat.dbf')
pd.set_option('display.width',200)
print(lc_vat.drop(columns='geometry', errors='ignore'))
print(prot_vat.drop(columns='geometry', errors='ignore'))
u,c = np.unique(lc[~np.isnan(lc)], return_counts=True)
print('LC values:', dict(zip(u.astype(int), c)))
u,c = np.unique(prot[~np.isnan(prot)], return_counts=True)
print('PROT values:', dict(zip(u.astype(int), c)))


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



import rasterio
from rasterio import features
from scipy import ndimage
from rasterio.transform import from_origin
tr = lc_meta['transform']; H,W = lc.shape
print('grid', H, W, tr)
print(hab[['OBJECTID','core','Name']])
# overlap check of habitats
from itertools import combinations
for a,b in combinations(range(len(hab)),2):
    print(a,b, round(hab.geometry.iloc[a].intersection(hab.geometry.iloc[b]).area,1))

# --- Ruggedness: Riley TRI over 3x3 window ---
z = elev.astype(float)
zp = np.pad(z, 1, mode='edge')
diffsq = np.zeros_like(z)
for dr in (-1,0,1):
    for dc in (-1,0,1):
        if dr==0 and dc==0: continue
        nb = zp[1+dr:1+dr+H, 1+dc:1+dc+W]
        diffsq += (nb - z)**2
tri = np.sqrt(diffsq)
p1,p99 = np.percentile(tri,[1,99])
print('TRI range', tri.min(), tri.max(), 'p1,p99', p1, p99, 'mean', tri.mean())
t = np.clip((tri-p1)/(p99-p1),0,1)
cost_rug = 10 - 9*t          # high ruggedness -> low cost

# --- Road distance ---
road_shapes = [(g,1) for g in roads.geometry if g is not None]
road_mask = features.rasterize(road_shapes, out_shape=(H,W), transform=tr, fill=0, all_touched=True, dtype='uint8').astype(bool)
print('road cells', road_mask.sum())
dist_road = ndimage.distance_transform_edt(~road_mask) * 30.0
print('road dist range', dist_road.min(), dist_road.max(), 'median', np.median(dist_road))
cost_road = 1 + 9*(1 - np.minimum(dist_road,1000)/1000.0)   # on road -> 10, >=1 km -> 1



# --- Land cover reclassification (NLCD code -> cost 1 easy .. 10 hard) ---
lc_cost_map = {11:8, 21:5, 22:7, 23:9, 24:10, 31:4, 41:1, 42:1, 43:1, 52:1,
               71:2, 81:5, 82:6, 90:2, 95:2}
lci = lc.astype(int)
cost_lc = np.full((H,W), np.nan)
for code,cval in lc_cost_map.items():
    cost_lc[lci==code] = cval
unk_lc = np.isin(lci,[0,255])
cost_lc[unk_lc] = 5.0        # unclassified cells: moderate value, flagged
print('LC unclassified (0/255) cells:', unk_lc.sum(), ' unmapped others:', np.isnan(cost_lc).sum()-unk_lc.sum())

# --- Protected status reclassification (1 very high .. 4 not protected) ---
pi = prot.astype(int)
prot_cost_map = {1:1, 2:3, 3:6, 4:10}
cost_prot = np.full((H,W), np.nan)
for code,cval in prot_cost_map.items():
    cost_prot[pi==code] = cval
no_status = np.isin(pi,[0,255])
cost_prot[no_status] = 10.0  # documented as "no protected status" -> treated as not protected
print('Protected no-status cells (0/255):', no_status.sum())

# --- Weighted sum ---
w = {'landcover':0.30, 'ruggedness':0.25, 'road':0.25, 'protected':0.20}
print('weights', w, 'sum', sum(w.values()))
cost_final = (w['landcover']*cost_lc + w['ruggedness']*cost_rug +
              w['road']*cost_road + w['protected']*cost_prot)
print('component ranges:')
for n,a in [('lc',cost_lc),('rug',cost_rug),('road',cost_road),('prot',cost_prot),('final',cost_final)]:
    print(f'  {n}: min={np.nanmin(a):.2f} max={np.nanmax(a):.2f} mean={np.nanmean(a):.2f} nan={np.isnan(a).sum()}')
print('final unique-ish count', len(np.unique(np.round(cost_final,2))))



from skimage.graph import MCP_Geometric
from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import combinations
reg = features.rasterize([(g, i+1) for i,g in enumerate(hab.geometry)], out_shape=(H,W), transform=tr, fill=0, dtype='int32')
names = {i+1: hab.Name.iloc[i] for i in range(len(hab))}
print('region cell counts:', {names[k]: int((reg==k).sum()) for k in names})
K = len(hab)
pair_cost = np.full((K,K), np.inf)
pair_path = {}
for i in range(1,K+1):
    starts = np.argwhere(reg==i)
    m = MCP_Geometric(cost_final, fully_connected=True)
    cum, _ = m.find_costs(starts=[tuple(s) for s in starts])
    for j in range(1,K+1):
        if j==i: continue
        cells = np.argwhere(reg==j)
        vals = cum[reg==j]
        k = np.argmin(vals)
        end = tuple(cells[k])
        pair_cost[i-1,j-1] = vals[k]
        path = np.array(m.traceback(end))
        pair_path[(i,j)] = path
        print(f'{names[i]} -> {names[j]}: cost={vals[k]:.1f}, path cells={len(path)}, length~{len(path)*30/1000:.1f} km')



import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, os, json
from shapely.geometry import LineString
os.makedirs('pred_results', exist_ok=True)
sym = np.where(np.isfinite(pair_cost), pair_cost, 0)
sym = np.maximum(sym, sym.T)  # symmetric
mst = minimum_spanning_tree(sym).toarray()
mst_edges = [(i+1,j+1) for i in range(K) for j in range(K) if mst[i,j]>0]
print('MST links:', [(names[a],names[b], round(sym[a-1,b-1],1)) for a,b in mst_edges])
def to_xy(p):
    return [(tr.c+(c+0.5)*tr.a, tr.f+(r+0.5)*tr.e) for r,c in p]
recs = []
for (a,b) in mst_edges:
    key = (a,b) if (a,b) in pair_path else (b,a)
    p = pair_path[key]
    recs.append({'from':names[a],'to':names[b],'cost':float(sym[a-1,b-1]),
                 'length_km':len(p)*30/1000, 'geometry':LineString(to_xy(p))})
gpd.GeoDataFrame(recs, geometry='geometry', crs=hab.crs).to_file('pred_results/optimal_connections.geojson', driver='GeoJSON')
pd.DataFrame(pair_cost, index=[names[i] for i in range(1,K+1)], columns=[names[i] for i in range(1,K+1)]).to_csv('pred_results/pair_cost_matrix.csv')
# save cost rasters
meta = dict(driver='GTiff', height=H, width=W, count=1, dtype='float32', crs=lc_meta['crs'], transform=tr, nodata=-9999)
with rasterio.open('pred_results/final_cost.tif','w',**meta) as dst: dst.write(cost_final.astype('float32'),1)
with rasterio.open('pred_results/cost_ruggedness.tif','w',**meta) as dst: dst.write(cost_rug.astype('float32'),1)
with rasterio.open('pred_results/cost_road.tif','w',**meta) as dst: dst.write(cost_road.astype('float32'),1)
with rasterio.open('pred_results/cost_landcover.tif','w',**meta) as dst: dst.write(cost_lc.astype('float32'),1)
with rasterio.open('pred_results/cost_protected.tif','w',**meta) as dst: dst.write(cost_prot.astype('float32'),1)
print(os.listdir('pred_results'))



extent = (tr.c, tr.c+W*30, tr.f-H*30, tr.f)
fig, axes = plt.subplots(1,2, figsize=(16,9), gridspec_kw={'width_ratios':[2.2,1]})
ax = axes[0]
im = ax.imshow(cost_final, extent=extent, cmap='RdYlGn_r', vmin=1, vmax=10, origin='upper', alpha=0.9)
hab.boundary.plot(ax=ax, color='k', linewidth=1.2)
for i,g in enumerate(hab.geometry):
    c = g.representative_point()
    ax.text(c.x, c.y, names[i+1], fontsize=10, fontweight='bold', ha='center', color='k',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
for (a,b) in mst_edges:
    key = (a,b) if (a,b) in pair_path else (b,a)
    p = np.array(to_xy(pair_path[key]))
    ax.plot(p[:,0], p[:,1], color='blue', linewidth=2.5)
ax.set_title('Final cost surface (1 = low cost, 10 = high cost)\nBlue: optimal region connections (minimum spanning network)')
ax.set_xlabel('Easting (m, UTM 11N)'); ax.set_ylabel('Northing (m)')
ax.ticklabel_format(style='plain'); ax.tick_params(labelsize=8)
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Cumulative cost class')
ax2 = axes[1]
pm = np.where(np.isfinite(pair_cost), pair_cost, np.nan)
im2 = ax2.imshow(pm, cmap='viridis')
lbl = [names[i] for i in range(1,K+1)]
ax2.set_xticks(range(K)); ax2.set_xticklabels(lbl, rotation=45, ha='right', fontsize=8)
ax2.set_yticks(range(K)); ax2.set_yticklabels(lbl, fontsize=8)
for i in range(K):
    for j in range(K):
        if i!=j: ax2.text(j,i,f'{pair_cost[i,j]:.0f}', ha='center', va='center', color='w', fontsize=9)
ax2.set_title('Least-cost path cost between regions\n(MST links: SS-LP, SS-SG, SS-SM)')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig('pred_results/optimal_path.png', dpi=150)
plt.close()
print('saved', os.path.getsize('pred_results/optimal_path.png'))



for (a,b) in mst_edges:
    key = (a,b) if (a,b) in pair_path else (b,a)
    p = pair_path[key]
    s, e = p[0], p[-1]
    print(f'{names[a]}-{names[b]}: start in {names[key[0]]}? {reg[s[0],s[1]]==key[0]}  end in {names[key[1]]}? {reg[e[0],e[1]]==key[1]}  '
          f'mean cell cost={cost_final[p[:,0],p[:,1]].mean():.2f}  cells={len(p)}')
print(os.listdir('pred_results'))
