# Correct: Horn slope/aspect, grouping by the 96 ScientificName taxa, bars of arithmetic means plus
# circular-mean dots on the aspect axes.
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


_src = rasterio.open('dataset/CatalinaBathymetry.tif')
bathy = _src.read().astype(float) if _src.count > 1 else _src.read(1).astype(float)
bathy_meta = dict(crs=_src.crs, transform=_src.transform, width=_src.width, height=_src.height,
               nodata=_src.nodata, dtype=_src.dtypes[0], count=_src.count, bounds=_src.bounds, res=_src.res)
if _src.nodata is not None:
    bathy[bathy == _src.nodata] = np.nan
print(f"Size: {_src.width} x {_src.height}, bands: {_src.count}, dtype: {_src.dtypes[0]}")
print(f"CRS: {_src.crs.to_string() if _src.crs else None}, resolution: {_src.res}")
print(f"Bounds: {tuple(round(float(_b), 4) for _b in _src.bounds)}")
print(f"NoData: {_src.nodata} (replaced by NaN in bathy)")
_valid = bathy[~np.isnan(bathy)]
print(f"Valid values: {_valid.size} of {bathy.size}" + (f", min={_valid.min():.4g}, max={_valid.max():.4g}, mean={_valid.mean():.4g}" if _valid.size else ""))
print("Variables: bathy = float array" + (" (bands, rows, cols)" if _src.count > 1 else " (rows, cols)") + ", bathy_meta = dict(crs, transform, nodata, ...)")
_src.close()



import numpy as np, geopandas as gpd, rasterio
z = np.pad(bathy, 1, mode='edge')
cs = bathy_meta['transform'].a
a,b,c = z[:-2,:-2], z[:-2,1:-1], z[:-2,2:]
d,f = z[1:-1,:-2], z[1:-1,2:]
g,h,i = z[2:,:-2], z[2:,1:-1], z[2:,2:]
dzdx = ((c+2*f+i)-(a+2*d+g))/(8*cs)
dzdy = ((g+2*h+i)-(a+2*b+c))/(8*cs)
slope = np.degrees(np.arctan(np.hypot(dzdx,dzdy)))
asp = np.degrees(np.arctan2(dzdy, -dzdx))
aspect = np.where(asp<0, 90-asp, np.where(asp>90, 360-asp+90, 90-asp))
aspect = np.where((dzdx==0)&(dzdy==0), -1, aspect)
print(np.nanmin(slope), np.nanmax(slope), np.nanmean(slope))
print(np.nanmin(aspect), np.nanmax(aspect), (aspect==-1).sum())
prof = dict(driver='GTiff', height=bathy.shape[0], width=bathy.shape[1], count=1, dtype='float32', crs=bathy_meta['crs'], transform=bathy_meta['transform'], nodata=-9999)
import os; os.makedirs('pred_results', exist_ok=True)
for name, arr in [('Slope',slope),('Aspect',aspect)]:
    with rasterio.open(f'pred_results/Catalina{name}.tif','w',**prof) as dst: dst.write(arr.astype('float32'),1)
pts = gpd.read_file('dataset/CoralandSpongeCatalina.geojson').to_crs(3310)
print(len(pts))



from rasterio.transform import rowcol
r, cidx = rowcol(bathy_meta['transform'], pts.geometry.x.values, pts.geometry.y.values)
r, cidx = np.array(r), np.array(cidx)
inside = (r>=0)&(r<bathy.shape[0])&(cidx>=0)&(cidx<bathy.shape[1])
print('inside:', inside.sum(), 'of', len(pts))
upd = pts.copy()
upd['Slope'] = np.nan; upd['Aspect'] = np.nan; upd['Elevation']=np.nan
upd.loc[inside,'Slope'] = slope[r[inside], cidx[inside]]
upd.loc[inside,'Aspect'] = aspect[r[inside], cidx[inside]]
upd.loc[inside,'Elevation'] = bathy[r[inside], cidx[inside]]
print(upd[['Slope','Aspect','Elevation','DepthInMeters']].describe())
print((upd.Aspect==-1).sum())



import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
upd.to_crs(4326).to_file('pred_results/CoralandSpongeCatalina_updated.geojson', driver='GeoJSON')
v = upd.copy(); v['AspValid'] = v.Aspect.where(v.Aspect>=0)
v['s'] = np.sin(np.radians(v.AspValid)); v['c'] = np.cos(np.radians(v.AspValid))
grp = v.groupby('ScientificName').agg(MeanSlope=('Slope','mean'), MeanAspect=('AspValid','mean'),
        s=('s','mean'), c=('c','mean'), Count=('Slope','size'))
grp['CircMeanAspect'] = (np.degrees(np.arctan2(grp.s, grp.c)) + 360) % 360
grp = grp.drop(columns=['s','c']).sort_values('MeanSlope')
grp.to_csv('pred_results/CoralandSponge_species_slope_aspect.csv')
print(len(grp)); print(grp.describe())
fig, axes = plt.subplots(1, 2, figsize=(16, 22), sharey=True)
y = np.arange(len(grp))
axes[0].barh(y, grp.MeanSlope, color='steelblue'); axes[0].set_xlabel('Mean slope (degrees)'); axes[0].set_title('Mean Slope by Species')
axes[1].barh(y, grp.MeanAspect, color='darkorange', label='Arithmetic mean aspect')
axes[1].scatter(grp.CircMeanAspect, y, color='k', s=12, zorder=3, label='Circular mean aspect')
axes[1].set_xlim(0, 360); axes[1].set_xticks(range(0,361,45)); axes[1].set_xlabel('Mean aspect (degrees clockwise from north)')
axes[1].set_title('Mean Aspect by Species'); axes[1].legend(loc='lower right')
axes[0].set_yticks(y); axes[0].set_yticklabels([f'{n} (n={k})' for n,k in zip(grp.index, grp.Count)], fontsize=7)
for ax in axes: ax.grid(axis='x', alpha=0.3); ax.set_ylim(-1, len(grp))
fig.suptitle('Mean Slope and Aspect of Coral and Sponge Species, Catalina Island', fontsize=15)
plt.tight_layout(rect=[0,0,1,0.98]); plt.savefig('pred_results/CoralandSponge.png', dpi=150); plt.close()
