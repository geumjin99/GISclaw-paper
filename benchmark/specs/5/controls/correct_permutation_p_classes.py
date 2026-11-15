# Correct: hexagon classes from permutation p-values (p_sim*2, FDR) drawn with color=; the classes are
# ordered in sign(z)*(1-p), not perfectly in z.
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


import geopandas as gpd
pts_raw = gpd.read_file("dataset/High_Blood_Level_Results.shp")
zips_raw = gpd.read_file("dataset/Sacramento_ZIP_Codes.shp")
print(len(pts_raw), pts_raw.crs, pts_raw.columns.tolist())
print(len(zips_raw), zips_raw.crs, zips_raw.columns.tolist())
points = pts_raw.to_crs(26910)
zips = zips_raw.to_crs(26910)
for n,g in [("points",points),("zips",zips)]:
    print(n, "invalid:", (~g.is_valid).sum(), "empty:", g.is_empty.sum(), "null:", g.geometry.isna().sum(), g.crs.to_epsg(), g.total_bounds)
if (~zips.is_valid).any():
    zips["geometry"] = zips.geometry.buffer(0)
study_area = gpd.GeoDataFrame(geometry=[zips.union_all() if hasattr(zips,'union_all') else zips.unary_union], crs=zips.crs)
print("study area km2:", study_area.area.iloc[0]/1e6, study_area.geom_type.iloc[0])
inside = points.within(study_area.geometry.iloc[0])
print("points inside:", inside.sum(), "of", len(points))
d = points[~inside].distance(study_area.geometry.iloc[0])
print("outside distances (m):", d.describe() if len(d) else "none")
print("duplicate locations:", points.geometry.duplicated().sum())


import numpy as np
from scipy.spatial import cKDTree
xy = np.column_stack([points.geometry.x, points.geometry.y])
dd, _ = cKDTree(xy).query(xy, k=2)
nn = dd[:,1]
print("NN dist mean/median/max:", nn.mean().round(1), np.median(nn).round(1), nn.max().round(1))
print("dup locations (nn=0):", (nn==0).sum())
print(study_area.total_bounds, study_area.area.iloc[0]/1e6, "km2")



import geopandas as gpd
from shapely.geometry import Polygon
R = 400.0  # side length / circumradius (m)
w = np.sqrt(3)*R  # flat-to-flat width
minx, miny, maxx, maxy = study_area.total_bounds
minx-=2*R; miny-=2*R; maxx+=2*R; maxy+=2*R
polys=[]; row=0; y=miny
angles = np.deg2rad(np.arange(30,390,60))  # pointy-top
while y < maxy:
    x = minx + (w/2 if row%2 else 0)
    while x < maxx:
        polys.append(Polygon(zip(x+R*np.cos(angles), y+R*np.sin(angles))))
        x += w
    y += 1.5*R; row += 1
grid = gpd.GeoDataFrame(geometry=polys, crs=points.crs)
hexes = grid[grid.intersects(study_area.geometry.iloc[0])].reset_index(drop=True)
hexes["hex_id"] = np.arange(len(hexes))
j = gpd.sjoin(points, hexes[["hex_id","geometry"]], how="left", predicate="within")
print("points unmatched:", j.hex_id.isna().sum(), "| joined rows:", len(j))
cnt = j.dropna(subset=["hex_id"]).groupby("hex_id").size()
hexes["count"] = hexes.hex_id.map(cnt).fillna(0).astype(int)
print("cell side R=%.0f m, width=%.0f m, area=%.3f km2" % (R, w, hexes.area.iloc[0]/1e6))
print("grid total:", len(grid), "| hexes kept:", len(hexes))
print("sum counts:", hexes['count'].sum(), "| max:", hexes['count'].max(), "| mean:", round(hexes['count'].mean(),3))
print("occupied:", (hexes['count']>0).sum(), "| mean per occupied:", round(hexes.loc[hexes['count']>0,'count'].mean(),2))
print(hexes['count'].value_counts().sort_index().to_dict())



import numpy as np, libpysal, esda
print(hexes.columns.tolist(), hexes.shape, hexes['count'].describe().to_dict())
cent = np.column_stack([hexes.geometry.centroid.x, hexes.geometry.centroid.y])
thr = libpysal.weights.min_threshold_distance(cent)
print("min threshold distance (all have >=1 neighbour):", thr)
# incremental spatial autocorrelation: Global Moran's I z across bands
res=[]
for b in np.arange(700, 5001, 300):
    wb = libpysal.weights.DistanceBand(cent, threshold=b, binary=True, silence_warnings=True)
    wb.transform='r'
    mi = esda.Moran(hexes['count'].values, wb, permutations=0)
    res.append((b, mi.I, mi.z_norm, wb.mean_neighbors, len(wb.islands)))
for r in res: print("band %5d  I=%.3f z=%.2f meanNb=%.1f islands=%d"%r)



band = 2500
w_gi = libpysal.weights.DistanceBand(cent, threshold=band, binary=True, silence_warnings=True)
print("islands:", len(w_gi.islands), "min/mean/max neighbours:", min(w_gi.cardinalities.values()), w_gi.mean_neighbors, max(w_gi.cardinalities.values()))
np.random.seed(42)
gi = esda.G_Local(hexes['count'].values.astype(float), w_gi, transform='B', permutations=9999, star=True, seed=42)
hexes['gi_z'] = gi.Zs
hexes['gi_p'] = gi.p_sim * 2  # two-sided pseudo p-value from one-sided p_sim
hexes['gi_p'] = hexes['gi_p'].clip(upper=1.0)
p = hexes['gi_p'].values
thr_fdr = {a: esda.fdr(p, a) for a in (0.10, 0.05, 0.01)}
print("FDR p thresholds:", thr_fdr)
def classify(z, pv):
    for a, lab in ((0.01,'99%'),(0.05,'95%'),(0.10,'90%')):
        if pv <= thr_fdr[a]:
            return ('Hot Spot - ' if z > 0 else 'Cold Spot - ') + lab + ' Confidence'
    return 'Not Significant'
hexes['hotspot_class'] = [classify(z, pv) for z, pv in zip(hexes.gi_z, hexes.gi_p)]
print("distance band used:", band, "m")
print(hexes['hotspot_class'].value_counts())
print(hexes.groupby('hotspot_class')['count'].agg(['mean','sum','size']))
print("gi_z range:", hexes.gi_z.min(), hexes.gi_z.max())


print(hexes.columns.tolist()); print(hexes['hotspot_class'].value_counts()); print(hexes[['count','gi_z','gi_p']].describe())

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
cls_order=['Hot Spot - 99% Confidence','Hot Spot - 95% Confidence','Hot Spot - 90% Confidence','Not Significant',
           'Cold Spot - 90% Confidence','Cold Spot - 95% Confidence','Cold Spot - 99% Confidence']
cls_col=dict(zip(cls_order,['#b2182b','#ef8a62','#fddbc7','#d9d9d9','#d1e5f0','#67a9cf','#2166ac']))
zips_l=zips.reset_index(drop=True).copy(); zips_l['zip_idx']=['ZIP polygon %d'%(i+1) for i in range(len(zips_l))]
fig,axes=plt.subplots(1,2,figsize=(16,10))
ax=axes[0]
hexes.plot(ax=ax,color=hexes['hotspot_class'].map(cls_col),edgecolor='white',linewidth=0.2)
zips_l.boundary.plot(ax=ax,color='black',linewidth=0.8)
for _,r_ in zips_l.iterrows():
    pt=r_.geometry.representative_point(); ax.annotate(r_.zip_idx.replace('ZIP polygon ','Z'),(pt.x,pt.y),fontsize=7,ha='center',color='black')
ax.legend(handles=[Patch(facecolor=cls_col[c],edgecolor='grey',label=f"{c} (n={int((hexes.hotspot_class==c).sum())})") for c in cls_order],
          loc='lower left',fontsize=8,title='Gi* hot spot class (FDR)',title_fontsize=9)
ax.set_title('Optimized Hot Spot Analysis (Getis-Ord Gi*)\n400 m hexagons, 2,500 m distance band',fontsize=11)
ax2=axes[1]
bounds=[-0.5,0.5,1.5,2.5,4.5,9.5]; cm=ListedColormap(['#f7f7f7','#fee8c8','#fdbb84','#e34a33','#7f0000']); nrm=BoundaryNorm(bounds,cm.N)
hexes.plot(ax=ax2,column='count',cmap=cm,norm=nrm,edgecolor='lightgrey',linewidth=0.2)
zips_l.boundary.plot(ax=ax2,color='black',linewidth=0.8)
ax2.legend(handles=[Patch(facecolor=cm(i),edgecolor='grey',label=l) for i,l in enumerate(['0','1','2','3–4','5–9'])],
           loc='lower left',fontsize=8,title='Cases per hexagon',title_fontsize=9)
ax2.set_title('Elevated blood lead cases aggregated to hexagons\n(individual locations not shown)',fontsize=11)
for a in axes:
    a.set_axis_off()
    x0,x1=a.get_xlim(); y0,y1=a.get_ylim()
    sx=x1-(x1-x0)*0.05-5000; sy=y0+(y1-y0)*0.04
    a.plot([sx,sx+5000],[sy,sy],color='k',lw=3); a.text(sx+2500,sy+300,'5 km',ha='center',fontsize=8)
    a.annotate('N',xy=(0.95,0.95),xytext=(0.95,0.87),xycoords='axes fraction',ha='center',fontsize=12,fontweight='bold',
               arrowprops=dict(facecolor='black',width=4,headwidth=10))
fig.suptitle('Children with Elevated Blood Lead Levels – Sacramento (privacy-preserving hexagon summary)',fontsize=14,fontweight='bold')
fig.text(0.5,0.02,'Data: High_Blood_Level_Results (n=373 cases), Sacramento ZIP codes. CRS: NAD83 / UTM 10N. Points aggregated; no individual locations displayed.',ha='center',fontsize=8)
plt.tight_layout(rect=[0,0.03,1,0.95])
fig_map=fig
# summary of ZIPs overlapping hot spots
hot=hexes[hexes.hotspot_class.str.startswith('Hot')]
ov=gpd.overlay(hot[['hex_id','count','geometry']],zips_l[['zip_idx','geometry']],how='intersection') if 'gpd' in dir() else None

import geopandas as gpd
ov=gpd.overlay(hot[['hex_id','count','hotspot_class','geometry']],zips_l[['zip_idx','geometry']],how='intersection')
ov['a']=ov.area
s=ov.groupby('zip_idx').agg(hot_hex=('hex_id','nunique'),hot_area_km2=('a',lambda v:v.sum()/1e6)).sort_values('hot_area_km2',ascending=False)
zarea=zips_l.set_index('zip_idx').area/1e6
s['pct_of_zip']=(100*s.hot_area_km2/zarea.reindex(s.index)).round(1)
print(s.round(2))
print('Hot-spot hexagons:',len(hot),'containing',int(hot['count'].sum()),'of',int(hexes['count'].sum()),'cases')
c=hot.dissolve().geometry.centroid.to_crs(4326).iloc[0]; print('Hot-spot cluster centroid (lon,lat): %.4f, %.4f'%(c.x,c.y))
print('Hot-spot extent (lon/lat):',hot.to_crs(4326).total_bounds.round(4))

import os
os.makedirs('pred_results', exist_ok=True)
print(hexes.columns.tolist(), hexes.geom_type.unique())
fig_map.savefig('pred_results/blood_lead_hotspots.png', dpi=300, bbox_inches='tight')
cols = [c for c in ['hex_id','count','gi_z','gi_p','hotspot_class'] if c in hexes.columns]
out = hexes[cols+['geometry']].copy()
for c in out.columns:
    if c!='geometry' and str(out[c].dtype).startswith('float'): out[c]=out[c].astype('float64')
out.to_crs(4326).to_file('pred_results/blood_lead_hex_hotspots.geojson', driver='GeoJSON')
chk = gpd.read_file('pred_results/blood_lead_hex_hotspots.geojson')
print(len(chk), chk['count'].sum(), chk['hotspot_class'].value_counts().to_dict())
for f in os.listdir('pred_results'): print(f, os.path.getsize('pred_results/'+f))