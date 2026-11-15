# Correct: ruggedness (5x5 std) and road distance both rescaled linearly 10 -> 1 (the task leaves the
# rescaling function open); corridors drawn with ax.plot.
import os, rasterio, numpy as np, pandas as pd, geopandas as gpd
os.makedirs('pred_results', exist_ok=True)

import rasterio, numpy as np, geopandas as gpd
def rd(p):
    with rasterio.open(p) as s:
        return s.read(1), s.profile.copy(), s.transform, s.crs
lc, prof, t1, c1 = rd('dataset/landCover.tif')
prot, _, t2, c2 = rd('dataset/Protected_Status.tif')
elev, _, t3, c3 = rd('dataset/Elevation.tif')
print(lc.shape, prot.shape, elev.shape)
print(t1==t2==t3, c1, c2, c3)
print(prof)
roads_gdf = gpd.read_file('dataset/Roads.geojson')
habitats_gdf = gpd.read_file('dataset/habitats.geojson')
print(roads_gdf.crs, habitats_gdf.crs, len(roads_gdf))
for n,a in [('lc',lc),('prot',prot)]:
    u,c = np.unique(a, return_counts=True); print(n, dict(zip(u.tolist(),c.tolist())))
print('elev', elev.min(), elev.max(), elev.mean(), len(np.unique(elev)), (elev==0).sum(), (elev==255).sum())
print(habitats_gdf[['OBJECTID','core','Name']])



import numpy as np
from scipy.ndimage import uniform_filter
print(elev.dtype, elev.min(), elev.max(), (elev==255).sum(), (elev==0).sum())
e = elev.astype('float64')
win = 5
m = uniform_filter(e, size=win, mode='nearest')
m2 = uniform_filter(e*e, size=win, mode='nearest')
rug = np.sqrt(np.clip(m2-m*m, 0, None))
print('rugged', rug.min(), rug.max(), rug.mean(), np.isnan(rug).sum())
# rescale 1-10, inverted: high ruggedness -> low cost
rmin, rmax = rug.min(), rug.max()
cost_elev = 10 - 9*(rug-rmin)/(rmax-rmin)
print('cost_elev', cost_elev.min(), cost_elev.max(), cost_elev.mean(), cost_elev.shape)
i_hi, i_lo = np.unravel_index(rug.argmax(), rug.shape), np.unravel_index(rug.argmin(), rug.shape)
print('check: max rug cost', cost_elev[i_hi], 'min rug cost', cost_elev[i_lo], 'corr', np.corrcoef(rug.ravel(), cost_elev.ravel())[0,1])


from rasterio import features
from scipy.ndimage import distance_transform_edt
tr = prof['transform'] if 'transform' in prof else t1
shape = lc.shape
road_mask = features.rasterize(((g,1) for g in roads_gdf.geometry if g is not None and not g.is_empty), out_shape=shape, transform=tr, fill=0, dtype='uint8', all_touched=True)
print("road cells", int(road_mask.sum()), "of", road_mask.size)
dist = distance_transform_edt(road_mask==0, sampling=30.0)
print("dist min/max (m)", dist.min(), dist.max(), "mean", dist.mean())
dmin, dmax = dist.min(), dist.max()
cost_road = 10 - 9*(dist-dmin)/(dmax-dmin)
print("cost_road", cost_road.min(), cost_road.max(), cost_road.mean(), np.isnan(cost_road).sum())
print("cost at road cells", cost_road[road_mask==1].mean(), "cost at farthest", cost_road[dist==dmax][0])
print(np.corrcoef(dist.ravel()[::50], cost_road.ravel()[::50])[0,1])

print(dict(zip(*[x.tolist() for x in np.unique(lc, return_counts=True)])))
print(dict(zip(*[x.tolist() for x in np.unique(prot, return_counts=True)])))

lc_map = {41:1,42:1,43:1,52:1,71:2,90:3,95:3,31:4,81:5,82:6,21:8,22:9,23:10,24:10,11:10,0:10,255:10}
prot_map = {1:1,2:3,3:6,4:8,0:10,255:10}
def reclass(a, mp):
    out = np.full(a.shape, np.nan)
    for k,v in mp.items(): out[a==k]=v
    return out
cost_lc = reclass(lc, lc_map); cost_prot = reclass(prot, prot_map)
print(np.isnan(cost_lc).sum(), np.isnan(cost_prot).sum())
print(np.unique(cost_lc), np.unique(cost_prot))


import numpy as np
from rasterio import features
w = {'lc':0.4,'prot':0.2,'road':0.2,'elev':0.2}
print(w, sum(w.values()))
for n_,a_ in [('lc',cost_lc),('prot',cost_prot),('road',cost_road),('elev',cost_elev)]:
    print(n_, a_.shape, np.isnan(a_).sum(), a_.min(), a_.max())
cost_final = w['lc']*cost_lc + w['prot']*cost_prot + w['road']*cost_road + w['elev']*cost_elev
print('final', np.isnan(cost_final).sum(), (cost_final<=0).sum(), cost_final.min(), cost_final.max(), cost_final.mean())
print(habitats_gdf[['core','Name']])
shapes = [(g, int(i)) for g,i in zip(habitats_gdf.geometry, habitats_gdf['core'])]
habitat_raster = features.rasterize(shapes, out_shape=cost_final.shape, transform=prof['transform'] if 'transform' in prof else t1, fill=0, dtype='uint8')
for i,nm in zip(habitats_gdf['core'], habitats_gdf['Name']):
    print(i, nm, int((habitat_raster==i).sum()))
print(np.unique(habitat_raster))



import numpy as np
from skimage.graph import MCP_Geometric
from scipy.sparse.csgraph import minimum_spanning_tree
ids=[1,2,3,4]
names={r.core:r.Name for r in habitats_gdf.itertuples()}
print(names, np.unique(habitat_raster,return_counts=True))
K=len(ids)
cost_mat=np.full((K,K),np.inf); np.fill_diagonal(cost_mat,0)
paths={}
for a_i,ia in enumerate(ids):
    src=np.argwhere(habitat_raster==ia)
    mcp=MCP_Geometric(cost_final,fully_connected=True)
    cum,_=mcp.find_costs([tuple(p) for p in src])
    for b_i,ib in enumerate(ids):
        if ib==ia: continue
        m_=habitat_raster==ib
        cells=np.argwhere(m_)
        vals=cum[m_]
        k=np.argmin(vals)
        end=tuple(cells[k])
        cost_mat[a_i,b_i]=vals[k]
        if a_i<b_i:
            paths[(ia,ib)]=np.array(mcp.traceback(end))
print(np.round(cost_mat,1))
sym=np.minimum(cost_mat,cost_mat.T)
print("asymmetry max",np.abs(cost_mat-cost_mat.T).max())
mst=minimum_spanning_tree(np.triu(sym,1)).toarray()
links=[(ids[i],ids[j],mst[i,j]) for i,j in zip(*np.nonzero(mst))]
path_mask=np.zeros(cost_final.shape,np.uint8)
selected={}
for ia,ib,c_ in links:
    p=paths[(min(ia,ib),max(ia,ib))]
    selected[(ia,ib)]=p
    path_mask[p[:,0],p[:,1]]=1
    print(names[ia],'<->',names[ib],'cost %.1f'%c_,'cells',len(p))
print(len(links),'links; path cells',path_mask.sum())


print(type(paths), list(paths.keys())[:5]); k=list(paths.keys())[0]; print(type(paths[k]), getattr(paths[k],'shape',None))
print(mst); print(names); print(cost_mat); print(path_mask.shape, path_mask.sum())

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, rasterio
from matplotlib.lines import Line2D
tr=t1
H,W=cost_final.shape
ext=[tr.c, tr.c+W*tr.a, tr.f+H*tr.e, tr.f]
links=[(int(i)+1,int(j)+1) for i,j in zip(*np.nonzero(mst))]
print(links)
fig,ax=plt.subplots(figsize=(13,11))
im=ax.imshow(cost_final,extent=ext,cmap='viridis',origin='upper')
plt.colorbar(im,ax=ax,shrink=0.7,label='Weighted cost (1 low – 10 high)')
habitats_gdf.boundary.plot(ax=ax,color='white',linewidth=1.5)
cols=['red','orange','magenta','cyan']
for (a_,b_),cc in zip(links,cols):
    p=paths[(a_,b_)]
    xs=tr.c+(p[:,1]+0.5)*tr.a; ys=tr.f+(p[:,0]+0.5)*tr.e
    ax.plot(xs,ys,color=cc,lw=2.2)
for _,r in habitats_gdf.iterrows():
    c=r.geometry.representative_point()
    ax.annotate(r['Name'],(c.x,c.y),color='white',fontsize=10,fontweight='bold',ha='center',
                xytext=(0,10),textcoords='offset points',bbox=dict(fc='black',alpha=0.5,pad=1))
handles=[Line2D([0],[0],color='white',lw=1.5,label='Habitat outline')]
for (a_,b_),cc in zip(links,cols):
    handles.append(Line2D([0],[0],color=cc,lw=2.2,label=f"{names[a_]} – {names[b_]} ({cost_mat[a_-1,b_-1]:.0f})"))
ax.legend(handles=handles,loc='lower left',title='Optimal links (accum. cost)',facecolor='lightgray')
ax.set_title('Optimal Region Connections between Mountain Lion Habitats')
ax.set_xlabel('Easting (m, EPSG:26911)'); ax.set_ylabel('Northing (m)')
plt.savefig('pred_results/optimal_path.png',dpi=150,bbox_inches='tight'); plt.close()
pr=prof.copy(); pr.update(dtype='float32',count=1,nodata=None)
with rasterio.open('pred_results/cost_final.tif','w',**pr) as d: d.write(cost_final.astype('float32'),1)
pr.update(dtype='uint8')
with rasterio.open('pred_results/optimal_path.tif','w',**pr) as d: d.write(path_mask.astype('uint8'),1)
import os; print([(f,os.path.getsize('pred_results/'+f)) for f in os.listdir('pred_results')])