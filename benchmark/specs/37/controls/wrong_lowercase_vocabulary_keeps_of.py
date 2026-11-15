"""Wrong: the lower-cased vocabulary keeps the unit fragment 'oF' -> 'of', so ordinary tweets count as
weather tweets.
"""
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


print(open('dataset/data20000.txt').read()[:500])
with open('dataset/12500-tweets-2021-09-24_01-48-23.json') as f:
    s=f.read(3000)
print(s)

import json, ast, re
txt=open('dataset/data20000.txt',encoding='utf-8').read().strip()
if not txt.startswith('{'): txt='{'+txt+'}'
vocab=ast.literal_eval(txt)
print(len(vocab), list(vocab.items())[:5])
tweets=json.load(open('dataset/12500-tweets-2021-09-24_01-48-23.json',encoding='utf-8'))
print(len(tweets))
import collections
print(collections.Counter((t.get('coordinates') is not None, t.get('place') is not None) for t in tweets))
print(tweets[0].keys())

from collections import defaultdict
import numpy as np
tmp=defaultdict(list)
for k,v in vocab.items():
    tmp[tuple(k.lower().replace('-','_').split('_'))].append(v)
lex={k:float(np.mean(v)) for k,v in tmp.items() if all(k)}
maxn=max(len(k) for k in lex); print(len(lex),maxn)
def get_text(t):
    if t.get('extended_tweet'): return t['extended_tweet'].get('full_text',t['text'])
    return t['text']
def score(text):
    text=re.sub(r'https?://\S+|@\w+','',text.lower())
    toks=re.findall(r"[a-z0-9']+",text)
    vals=[];i=0
    while i<len(toks):
        hit=False
        for n in range(min(maxn,len(toks)-i),0,-1):
            g=tuple(toks[i:i+n])
            if g in lex: vals.append(lex[g]); i+=n; hit=True; break
        if not hit: i+=1
    return vals
rows=[]
for t in tweets:
    vals=score(get_text(t))
    if not vals: continue
    if t.get('coordinates'):
        lon,lat=t['coordinates']['coordinates']
    elif t.get('place') and t['place'].get('bounding_box'):
        c=np.array(t['place']['bounding_box']['coordinates'][0]); lon,lat=c[:,0].mean(),c[:,1].mean()
    else: continue
    rows.append(dict(id=t['id'],text=get_text(t),lon=lon,lat=lat,value=sum(vals),nterms=len(vals),ptype=t['place']['place_type'] if t.get('place') else 'point'))
import pandas as pd
df=pd.DataFrame(rows); print(len(df)); print(df.ptype.value_counts()); print(df[['text','value']].head(10).to_string())

import geopandas as gpd
cty=gpd.read_file('dataset/cb_2020_us_county_20m.shp'); print(cty.crs, len(cty), cty.columns.tolist())
has_pt=df.id.isin([t['id'] for t in tweets if t.get('coordinates')])
d=df[has_pt | df.ptype.isin(['city','poi','neighborhood'])].copy()
g=gpd.GeoDataFrame(d,geometry=gpd.points_from_xy(d.lon,d.lat),crs='EPSG:4326').to_crs(cty.crs)
j=gpd.sjoin(g,cty[['GEOID','NAME','STUSPS','geometry']],predicate='within')
print(len(d),len(j))
agg=j.groupby('GEOID').agg(n=('value','size'),total=('value','sum')).reset_index()
agg['heat']=agg.total/agg.n
agg['heat_norm']=(agg.heat-agg.heat.min())/(agg.heat.max()-agg.heat.min())
print(len(agg)); print(agg.describe())

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, mapclassify
m=cty.merge(agg,on='GEOID',how='left')
print(m[m.heat.notna()].STUSPS.isin(['AK','HI','PR']).sum())
conus=m[~m.STUSPS.isin(['AK','HI','PR'])].to_crs(5070)
nb=mapclassify.NaturalBreaks(agg.heat_norm,k=5); print(nb)
fig,ax=plt.subplots(figsize=(16,10))
conus.plot(ax=ax,color='#e8e8e8',edgecolor='white',linewidth=0.2)
conus[conus.heat_norm.notna()].plot(ax=ax,column='heat_norm',scheme='UserDefined',classification_kwds={'bins':list(nb.bins)},
    cmap='YlOrRd',edgecolor='grey',linewidth=0.3,legend=True,
    legend_kwds={'title':'Normalized heat exposure\n(Natural Breaks)','loc':'lower right','fmt':'{:.2f}'})
conus.dissolve('STUSPS').boundary.plot(ax=ax,color='black',linewidth=0.4)
ax.set_title('US County Heat Exposure from Twitter Sentiment (2021-09-25)\nMean heat-vocabulary score per tweet, min-max normalized; grey = no data',fontsize=14)
ax.set_axis_off(); plt.tight_layout()
plt.savefig('pred_results/US_heat_exposure_map.png',dpi=200); plt.close()
agg.merge(cty[['GEOID','NAME','STUSPS']],on='GEOID').to_csv('pred_results/county_heat_exposure.csv',index=False)
import os; print(os.listdir('pred_results'))