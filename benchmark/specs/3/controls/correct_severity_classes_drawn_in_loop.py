# Correct: dNBR classified into USGS severity classes, vectorised, and drawn one class per loop
# iteration (sub = gdf[gdf['class']==k]; sub.plot), so the mapped layer is the union of the subsets.
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


import rasterio, numpy as np
for f in ['dataset/G_2014.tif','dataset/G_2015.tif']:
    with rasterio.open(f) as s:
        print(s.count, s.shape, s.crs, s.dtypes[0], s.nodata, s.bounds, s.res)


def nbr(f):
    with rasterio.open(f) as s:
        b5=s.read(5).astype(float); b7=s.read(7).astype(float); tr=s.transform; crs=s.crs
    m=(b5==65536)|(b7==65536)|((b5+b7)==0)
    n=np.where(m,np.nan,(b5-b7)/(b5+b7))
    return n,tr,crs
n14,tr,crs=nbr('dataset/G_2014.tif'); n15,_,_=nbr('dataset/G_2015.tif')
dnbr=n14-n15
print(np.nanpercentile(n14,[1,50,99]),np.nanpercentile(n15,[1,50,99]),np.nanpercentile(dnbr,[1,50,90,99,99.9]))
for t in [0.1,0.27,0.44,0.66]: print(t,(dnbr>t).sum())

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, os
os.makedirs('pred_results',exist_ok=True)
fig,ax=plt.subplots(1,3,figsize=(18,6))
ax[0].imshow(n14,vmin=-0.2,vmax=0.6);ax[1].imshow(n15,vmin=-0.2,vmax=0.6);im=ax[2].imshow(dnbr,vmin=-0.3,vmax=0.7,cmap='RdYlGn_r');plt.colorbar(im)
plt.savefig('/tmp/check.png',dpi=60)

from scipy import ndimage
for t in [0.1,0.2,0.27]:
    lab,n=ndimage.label(np.nan_to_num(dnbr)>t)
    sz=np.bincount(lab.ravel())[1:]; sz.sort()
    print(t,n,sz[-10:])

import geopandas as gpd
from rasterio.features import shapes, sieve
from shapely.geometry import shape
bins=[0.1,0.27,0.44,0.66]
cls=np.digitize(np.nan_to_num(dnbr,nan=-1),bins).astype('int32')  # 0 unburned,1 low,2 mod-low,3 mod-high,4 high
cls=ndimage.median_filter(cls,size=3).astype('int32')
burn=(cls>0).astype('uint8')
burn=sieve(burn,size=11,connectivity=8)  # remove patches < ~1 ha
cls=np.where(burn==1,np.maximum(cls,1),0).astype('int32')
names={1:'Low',2:'Moderate-low',3:'Moderate-high',4:'High'}
recs=[(shape(g),int(v)) for g,v in shapes(cls,mask=cls>0,transform=tr)]
gdf=gpd.GeoDataFrame({'class':[v for _,v in recs]},geometry=[g for g,_ in recs],crs=crs)
gdf['severity']=gdf['class'].map(names); gdf['area_ha']=gdf.area/1e4
gdf.to_file('pred_results/burn_scars.geojson',driver='GeoJSON')
summ=gdf.groupby('severity')['area_ha'].sum(); print(summ, summ.sum(), len(gdf))

from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
with rasterio.open('dataset/G_2015.tif') as s:
    rgb=np.dstack([s.read(b).astype(float) for b in (7,5,4)])
rgb[rgb>=65536]=np.nan
for i in range(3):
    lo,hi=np.nanpercentile(rgb[...,i],[2,98]); rgb[...,i]=np.clip((rgb[...,i]-lo)/(hi-lo),0,1)
rgb=np.nan_to_num(rgb)
ext=[tr.c, tr.c+tr.a*n14.shape[1], tr.f+tr.e*n14.shape[0], tr.f]
fig,axs=plt.subplots(2,2,figsize=(16,15))
for a,d,t in [(axs[0,0],n14,'NBR 2014'),(axs[0,1],n15,'NBR 2015')]:
    im=a.imshow(d,cmap='RdYlGn',vmin=-0.3,vmax=0.7,extent=ext); a.set_title(t,fontsize=14); plt.colorbar(im,ax=a,shrink=0.8,label='NBR')
im=axs[1,0].imshow(dnbr,cmap='RdYlBu_r',vmin=-0.3,vmax=0.7,extent=ext)
axs[1,0].set_title('dNBR (NBR 2014 − NBR 2015)',fontsize=14); plt.colorbar(im,ax=axs[1,0],shrink=0.8,label='dNBR')
a=axs[1,1]; a.imshow(rgb,extent=ext)
cols={1:'#ffff66',2:'#ffa500',3:'#ff3300',4:'#8b008b'}
for k,c in cols.items():
    sub=gdf[gdf['class']==k]
    if len(sub): sub.plot(ax=a,facecolor=c,edgecolor=c,linewidth=0.2,alpha=0.75)
h=[mpatches.Patch(color=cols[k],label=f"{names[k]} ({summ.get(names[k],0):,.0f} ha)") for k in cols]
a.legend(handles=h,title='Burn severity (USGS dNBR)',loc='lower right',fontsize=10)
a.set_title(f'Burn scar polygons over 2015 false colour (B7-B5-B4)\nTotal burned: {summ.sum():,.0f} ha',fontsize=14)
for a in axs.ravel(): a.set_xlabel('Easting (m, UTM 12N)'); a.set_ylabel('Northing (m)'); a.ticklabel_format(style='plain'); a.tick_params(labelsize=8)
fig.suptitle('Burn Scar Analysis using Landsat 8 Normalized Burn Ratio (2014 vs 2015)',fontsize=18)
plt.tight_layout(); plt.savefig('pred_results/burn_scar_analysis.png',dpi=150); print('ok')