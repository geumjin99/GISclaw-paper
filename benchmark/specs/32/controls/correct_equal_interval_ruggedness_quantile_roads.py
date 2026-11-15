# Correct: ruggedness rescaled by 10 equal-interval classes and road distance by 10 quantile classes
# (not MSLarge).
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



import rasterio, numpy as np, geopandas as gpd
arrs={}; metas={}
for k,p in [('elev','dataset/Elevation.tif'),('lc','dataset/landCover.tif'),('prot','dataset/Protected_Status.tif')]:
    with rasterio.open(p) as src:
        arrs[k]=src.read(1); metas[k]=(src.shape,src.transform,src.crs,src.nodata,src.dtypes[0])
    print(k, metas[k])
elev,lc,prot=arrs['elev'],arrs['lc'],arrs['prot']
shape,transform,crs=metas['elev'][0],metas['elev'][1],metas['elev'][2]
print("same grid:", all(metas[k][0]==shape and metas[k][1]==transform and metas[k][2]==crs for k in metas))
print("elev range", elev.min(), elev.max(), elev.mean())
for name,a in [('landCover',lc),('Protected_Status',prot)]:
    u,c=np.unique(a,return_counts=True); print(name, dict(zip(u.tolist(),c.tolist())))



roads=gpd.read_file('dataset/Roads.geojson'); habitats=gpd.read_file('dataset/habitats.geojson')
print(roads.crs.to_epsg(), habitats.crs.to_epsg(), crs.to_epsg())
if roads.crs!=crs: roads=roads.set_crs(crs, allow_override=True)
if habitats.crs!=crs: habitats=habitats.set_crs(crs, allow_override=True)
print(len(roads), roads.geom_type.value_counts().to_dict(), roads.total_bounds)
print(habitats[['OBJECTID','core','Name','Shape_Area']]); print(habitats.total_bounds)
print("elevation 0 cells:", (elev==0).sum(), " 255 cells:", (elev==255).sum())



import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter, distance_transform_edt
from rasterio.features import rasterize
e = elev.astype(np.float32)
frange = maximum_filter(e, size=3) - minimum_filter(e, size=3)
print("focal range min/max/mean:", frange.min(), frange.max(), frange.mean())
print("percentiles 10..100:", np.percentile(frange, np.arange(10,101,10)))
road_r = rasterize(((g,1) for g in roads.geometry if g is not None and not g.is_empty),
                   out_shape=shape, transform=transform, fill=0, all_touched=True, dtype='uint8')
print("road cells:", road_r.sum())
dist = distance_transform_edt(road_r == 0) * 30.0
print("dist min/max/mean:", dist.min(), dist.max(), dist.mean())
print("dist percentiles:", np.percentile(dist, np.arange(10,101,10)))



# Ruggedness: equal interval, 10 classes; class 1 = flattest -> cost 10, class 10 = most rugged -> cost 1
edges_r = np.linspace(frange.min(), frange.max(), 11)
rug_class = np.clip(np.digitize(frange, edges_r[1:-1], right=True) + 1, 1, 10)
rugged_cost = (11 - rug_class).astype(np.int16)
# Road distance: quantile classes; class 1 = nearest -> cost 10
edges_d = np.unique(np.percentile(dist, np.arange(10, 100, 10)))
d_class = np.clip(np.digitize(dist, edges_d, right=True) + 1, 1, 10)
road_cost = (11 - d_class).astype(np.int16)
print("rugged edges:", np.round(edges_r,2))
print("rugged_cost min/max:", rugged_cost.min(), rugged_cost.max(), "hist:", dict(zip(*np.unique(rugged_cost, return_counts=True))))
print("dist edges:", np.round(edges_d,1))
print("road_cost min/max:", road_cost.min(), road_cost.max(), "hist:", dict(zip(*np.unique(road_cost, return_counts=True))))
print("mean dist by road_cost:", {int(v): round(float(dist[road_cost==v].mean()),1) for v in np.unique(road_cost)})
print("mean frange by rugged_cost:", {int(v): round(float(frange[rugged_cost==v].mean()),2) for v in np.unique(rugged_cost)})
print(rugged_cost.shape, road_cost.shape)


import numpy as np
for nm, a_ in [('lc', lc), ('prot', prot)]:
    u_, c_ = np.unique(a_, return_counts=True)
    print(nm, dict(zip(u_.tolist(), c_.tolist())))

lc_map = {11:10, 21:8, 22:9, 23:10, 24:10, 31:6, 41:1, 42:1, 43:1, 52:2,
          71:5, 81:7, 82:7, 90:5, 95:5, 0:10, 255:10}
lc_cost = np.full(lc.shape, 10, dtype=np.int16)  # unmapped / no-data -> max cost
for k_, v_ in lc_map.items():
    lc_cost[lc == k_] = v_
prot_map = {1:1, 2:3, 3:6, 4:9, 0:10, 255:10}
prot_cost = np.full(prot.shape, 10, dtype=np.int16)
for k_, v_ in prot_map.items():
    prot_cost[prot == k_] = v_
print('lc_map', lc_map); print('prot_map', prot_map)
for nm, a_ in [('lc_cost', lc_cost), ('prot_cost', prot_cost)]:
    u_, c_ = np.unique(a_, return_counts=True)
    print(nm, a_.shape, a_.dtype, dict(zip(u_.tolist(), c_.tolist())))
print('unmapped lc values:', set(np.unique(lc).tolist()) - set(lc_map))

import numpy as np
cost_surface = (1.25*rugged_cost.astype(np.float64) + 1.0*road_cost + 1.25*lc_cost + 1.0*prot_cost)
print(cost_surface.shape, cost_surface.dtype)
print("NaN:", np.isnan(cost_surface).sum(), "nonpositive:", (cost_surface<=0).sum())
print("min/max/mean:", cost_surface.min(), cost_surface.max(), round(cost_surface.mean(),3))
print("percentiles 5/25/50/75/95:", np.percentile(cost_surface,[5,25,50,75,95]))


import numpy as np, networkx as nx, geopandas as gpd, time
from rasterio.features import rasterize
from skimage.graph import MCP_Geometric
from shapely.geometry import LineString
hab_r = rasterize([(g, int(c)) for g, c in zip(habitats.geometry, habitats.core)], out_shape=shape, transform=transform, fill=0, dtype='int32')
names = dict(zip(habitats.core.astype(int), habitats.Name))
print({c:(names[c], int((hab_r==c).sum())) for c in names})
cores = sorted(names)
pair_costs, pair_paths = {}, {}
for i_, ci in enumerate(cores):
    t=time.time()
    mcp = MCP_Geometric(cost_surface, fully_connected=True)
    starts = np.argwhere(hab_r==ci)
    cum, _ = mcp.find_costs(starts)
    for cj in cores[i_+1:]:
        cells = np.argwhere(hab_r==cj)
        vals = cum[cells[:,0], cells[:,1]]
        k = np.argmin(vals); end = tuple(cells[k])
        pair_costs[(ci,cj)] = float(vals[k])
        pair_paths[(ci,cj)] = np.array(mcp.traceback(end))
    print(ci, names[ci], f"{time.time()-t:.1f}s")
for k,v in pair_costs.items(): print(names[k[0]], '-', names[k[1]], round(v,1), len(pair_paths[k]))



G = nx.Graph()
for (a_i,b_i),v in pair_costs.items(): G.add_edge(a_i,b_i,weight=v)
mst_edges = sorted([tuple(sorted((u,w))) for u,w in nx.minimum_spanning_tree(G).edges()])
recs=[]
for e_ in mst_edges:
    rc = pair_paths[e_]
    xs, ys = transform * (rc[:,1]+0.5, rc[:,0]+0.5)
    recs.append(dict(from_core=e_[0], to_core=e_[1], from_name=names[e_[0]], to_name=names[e_[1]],
                     acc_cost=pair_costs[e_], n_cells=len(rc), geometry=LineString(zip(xs,ys))))
paths_gdf = gpd.GeoDataFrame(recs, crs=crs)
paths_gdf['length_m'] = paths_gdf.length
print(paths_gdf.drop(columns='geometry'))
# check endpoints lie in correct habitats
for _,r in paths_gdf.iterrows():
    rc = pair_paths[(r.from_core,r.to_core)]
    print(r.from_name, hab_r[tuple(rc[0])], hab_r[tuple(rc[-1])], 'mean cost along path', cost_surface[rc[:,0],rc[:,1]].mean().round(2))


print('paths_gdf' in globals()); print(recs[:1] if recs else recs); print(mst_edges); print(names)

import os, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
os.makedirs('pred_results', exist_ok=True)
print(paths_gdf[['from_name','to_name','acc_cost','n_cells']], paths_gdf.length.round(0).tolist())
l,b,r,t_=307049.4382, 3763070.3523, 398909.4382, 3837890.3523
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(cost_surface, extent=(l,r,b,t_), cmap='viridis', origin='upper')
cb = fig.colorbar(im, ax=ax, shrink=0.7); cb.set_label('Weighted cost (higher = harder to traverse)')
habitats.boundary.plot(ax=ax, color='white', linewidth=2)
for _, row in habitats.iterrows():
    pt = row.geometry.representative_point()
    ax.annotate(row['Name'], (pt.x, pt.y), color='white', fontsize=11, fontweight='bold', ha='center',
                xytext=(0,12), textcoords='offset points', bbox=dict(fc='black', alpha=0.5, ec='none'))
paths_gdf.plot(ax=ax, color='red', linewidth=2.5)
ax.legend(handles=[Line2D([0],[0],color='white',lw=2,label='Mountain lion habitat'),
                   Line2D([0],[0],color='red',lw=2.5,label='Optimal connection (least-cost path)')],
          loc='lower left', facecolor='lightgray')
ax.set_title('Optimal Region Connections between Mountain Lion Habitats\n(cost = 1.25·ruggedness + 1·road dist + 1.25·land cover + 1·protection)')
ax.set_xlabel('Easting (m, UTM 11N)'); ax.set_ylabel('Northing (m)')
plt.savefig('pred_results/optimal_path.png', dpi=200, bbox_inches='tight'); plt.close()
print(os.listdir('pred_results'), os.path.getsize('pred_results/optimal_path.png'))