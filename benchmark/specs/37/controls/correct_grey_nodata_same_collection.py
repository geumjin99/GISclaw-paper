"""Correct: all counties are drawn in one collection with color=...; no-data counties are grey
(0.85, 0.85, 0.85), which is not on the YlOrRd colormap.
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


with open('dataset/data20000.txt', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()
print(len(lines)); print(lines[:5]); print(lines[-3:])
heat_dict = {}; bad = 0; dup = 0
for ln in lines:
    ln = ln.strip()
    if not ln or ':' not in ln:
        bad += 1; continue
    term, val = ln.rsplit(':', 1)
    term = term.strip().strip('"\'').lower()
    try:
        v = float(val.strip().strip(',').strip('"\''))
    except ValueError:
        bad += 1; continue
    if not term:
        bad += 1; continue
    if term in heat_dict: dup += 1
    heat_dict[term] = v
import numpy as np
vals = np.array(list(heat_dict.values()))
print('size', len(heat_dict), 'malformed', bad, 'dups', dup)
print(list(heat_dict.items())[:10])
print('range', vals.min(), vals.max(), 'mean', vals.mean())
print('multiword terms:', sum(' ' in k for k in heat_dict))

import re
raw = lines[0]
print(repr(raw[:80])); print(repr(raw[-200:]))
pat = re.compile(r"""(['"])(.*?)\1\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)""")
matches = pat.findall(raw)
heat_dict = {}; dup = 0
for _, t, v in matches:
    t = t.strip().lower()
    if t in heat_dict: dup += 1
    heat_dict[t] = float(v)
print('pairs matched', len(matches), 'dict size', len(heat_dict), 'case-dup merged', dup)
print('approx entries by colon count:', raw.count("': "))

import ast
from collections import defaultdict
raw_d = ast.literal_eval('{' + raw.strip().rstrip(',') + '}')
print('raw entries', len(raw_d))
acc = defaultdict(list)
for k, v in raw_d.items():
    acc[str(k).strip().lower()].append(float(v))
heat_dict = {k: float(np.mean(v)) for k, v in acc.items() if k}
n_merged = sum(len(v) > 1 for v in acc.values())
vals = np.array(list(heat_dict.values()))
print('heat_dict size', len(heat_dict), '| terms with case-variants averaged:', n_merged)
print('score range', vals.min().round(4), vals.max().round(4), 'mean', vals.mean().round(4), 'NaN', np.isnan(vals).sum())
print('multi-word (underscore) terms:', sum('_' in k for k in heat_dict))
print(list(heat_dict.items())[:8])
for t in ['sweltering','heatwave','hot_humid','sunny','chill']: print(t, heat_dict.get(t))

p='dataset/12500-tweets-2021-09-24_01-48-23.json'
with open(p) as fh: head=fh.read(2000)
print(head[:2000])

import json, pandas as pd, numpy as np
with open(p) as fh: tweets_raw=json.load(fh)
print(type(tweets_raw), len(tweets_raw))
t0=tweets_raw[0]
print(list(t0.keys()))
print('coordinates',t0.get('coordinates'),'geo',t0.get('geo'))
print('place',json.dumps(t0.get('place'))[:600])
print('n coords:',sum(1 for t in tweets_raw if t.get('coordinates')),
      'n place bbox:',sum(1 for t in tweets_raw if t.get('place') and t['place'].get('bounding_box')),
      'n extended:',sum(1 for t in tweets_raw if 'extended_tweet' in t),
      'n full_text:',sum(1 for t in tweets_raw if 'full_text' in t))
from collections import Counter
print(Counter(t['place']['country_code'] for t in tweets_raw if t.get('place')).most_common(5))
print(Counter(t['place']['place_type'] for t in tweets_raw if t.get('place')).most_common())

rows=[]
for t in tweets_raw:
    text = t['extended_tweet'].get('full_text', t['text']) if 'extended_tweet' in t else t.get('full_text', t.get('text'))
    lon=lat=np.nan; src=None
    c=t.get('coordinates')
    if c and c.get('type')=='Point':
        lon,lat=c['coordinates']; src='point'
    else:
        pl=t.get('place') or {}
        bb=pl.get('bounding_box')
        if bb and bb.get('coordinates'):
            ring=np.array(bb['coordinates'][0],dtype=float)
            lon,lat=(ring[:,0].min()+ring[:,0].max())/2,(ring[:,1].min()+ring[:,1].max())/2; src='bbox_centroid'
    pl=t.get('place') or {}
    rows.append(dict(id=t['id_str'],text=text,lon=lon,lat=lat,geo_source=src,
                     place_type=pl.get('place_type'),country_code=pl.get('country_code')))
tweets_df=pd.DataFrame(rows)
print('total tweets:',len(tweets_df))
print('geolocated:',tweets_df[['lon','lat']].notna().all(axis=1).sum())
print(tweets_df.geo_source.value_counts(dropna=False))
print('US geolocated:',((tweets_df.country_code=='US')&tweets_df.lon.notna()).sum())
print(tweets_df[['lon','lat']].describe().loc[['min','max']])
print(tweets_df.head(3))

import collections
keys=list(heat_dict)
print(len(keys), keys[:10])
nus=sum('_' in k for k in keys); nsp=sum(' ' in k for k in keys)
print('underscore terms',nus,'space terms',nsp)
print(collections.Counter(k.count('_')+1 for k in keys))
print(tweets_df.columns.tolist()); print(tweets_df.head(3)); print(tweets_df[['lon','lat']].isna().sum())

import re, numpy as np, geopandas as gpd
chars=collections.Counter(ch for k in keys for ch in k if not ch.isalnum() and ch!='_'); print('odd chars in terms:',chars.most_common(10))
tok_re=re.compile(r"[a-z0-9']+")
MAXN=4
def match_terms(text):
    toks=[t.strip("'") for t in tok_re.findall(str(text).lower().replace('’',"'"))]
    toks=[t for t in toks if t]
    i=0; hits=[]
    while i<len(toks):
        for n in range(min(MAXN,len(toks)-i),0,-1):
            cand='_'.join(toks[i:i+n])
            if cand in heat_dict:
                hits.append(cand); i+=n; break
        else: i+=1
    return hits
tw=tweets_df.copy()
tw['terms']=tw['text'].map(match_terms)
tw['n_terms']=tw['terms'].str.len()
tw['score']=tw['terms'].map(lambda h: float(sum(heat_dict[x] for x in h)))
print('tweets with >=1 match:',(tw.n_terms>0).sum(),'of',len(tw))
print(collections.Counter(x for h in tw.terms for x in h).most_common(20))

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SW
print({k:heat_dict[k] for k in ['of','c'] })
sw_in=[k for k in heat_dict if k in SW]; short=[k for k in heat_dict if len(k)<=2]
print('stopword terms:',len(sw_in),sw_in[:40]); print('<=2 char terms:',len(short),short[:40])
print(tw.loc[tw.terms.map(lambda h:'c' in h),'text'].head(5).tolist())

NOISE={'of','c'}
url_re=re.compile(r"https?://\S+|@\w+")
def match_terms2(text):
    s=url_re.sub(' ',str(text).lower().replace('’',"'"))
    toks=[t.strip("'") for t in tok_re.findall(s)]; toks=[t for t in toks if t]
    i=0; hits=[]
    while i<len(toks):
        for n in range(min(MAXN,len(toks)-i),0,-1):
            cand='_'.join(toks[i:i+n])
            if cand in heat_dict and not (n==1 and cand in NOISE):
                hits.append(cand); i+=n; break
        else: i+=1
    return hits
tw['terms']=tw['text'].map(match_terms2); tw['n_terms']=tw.terms.str.len()
tw['score']=tw.terms.map(lambda h: float(sum(heat_dict[x] for x in h)))
print('matched:',(tw.n_terms>0).sum()); print(tw.loc[tw.n_terms>0,'place_type'].value_counts(dropna=False))
print(tw.loc[tw.n_terms>0,'geo_source'].value_counts())
print(collections.Counter(x for h in tw.terms for x in h).most_common(15))

keep=(tw.n_terms>0)&tw.lon.notna()&tw.lat.notna()&~tw.place_type.isin(['country','admin'])
wsel=tw.loc[keep,['id','text','score','n_terms','lon','lat','geo_source','place_type','country_code']].reset_index(drop=True)
wsel['terms']=tw.loc[keep,'terms'].map(lambda h:'|'.join(h)).values
weather_gdf=gpd.GeoDataFrame(wsel,geometry=gpd.points_from_xy(wsel.lon,wsel.lat),crs='EPSG:4269')
print('weather tweets:',len(weather_gdf)); print(weather_gdf.country_code.value_counts().head())
print(weather_gdf.score.describe()); print(weather_gdf.n_terms.value_counts())
print(weather_gdf[['text','terms','score']].head(5).to_string(max_colwidth=60))

import geopandas as gpd
counties = gpd.read_file("dataset/cb_2020_us_county_20m.shp")
print(len(counties), counties.crs, weather_gdf.crs)
pts = weather_gdf.to_crs(counties.crs) if weather_gdf.crs != counties.crs else weather_gdf
joined = gpd.sjoin(pts, counties[['GEOID','NAME','STUSPS','geometry']], how='left', predicate='within')
print("points:", len(pts), "joined rows:", len(joined), "matched:", joined['GEOID'].notna().sum())
print(joined.loc[joined['GEOID'].isna(), ['text']].head())
m = joined.dropna(subset=['GEOID'])
county_stats = m.groupby('GEOID').agg(tweet_count=('score','size'), score_sum=('score','sum'),
                                      NAME=('NAME','first'), STUSPS=('STUSPS','first')).reset_index()
county_stats['heat_exposure'] = county_stats['score_sum']/county_stats['tweet_count']
print("counties with tweets:", len(county_stats))
print(county_stats['heat_exposure'].describe())
print(county_stats.sort_values('heat_exposure', ascending=False).head(5))
print(county_stats.sort_values('tweet_count', ascending=False).head(5))

um = joined[joined['GEOID'].isna()]
print(um.geometry.x.describe()[['min','max']], um.geometry.y.describe()[['min','max']])
us_bbox = um.cx[-125:-66, 24:50]
print("unmatched inside CONUS bbox:", len(us_bbox))
print(us_bbox[['text']].head())

cs = county_stats.copy()
mn, mx = cs['heat_exposure'].min(), cs['heat_exposure'].max()
cs['heat_norm'] = 0.0 if mx == mn else (cs['heat_exposure'] - mn) / (mx - mn)
counties_merged = counties.merge(cs[['GEOID','tweet_count','score_sum','heat_exposure','heat_norm']], on='GEOID', how='left')
print(len(counties_merged), counties_merged['heat_norm'].notna().sum(), counties_merged.crs)
print('raw range', mn, mx)
print('norm range', counties_merged['heat_norm'].min(), counties_merged['heat_norm'].max(), 'unique', counties_merged['heat_norm'].nunique())
print(counties_merged.dropna(subset=['heat_norm']).sort_values('heat_norm', ascending=False)[['GEOID','NAME','STUSPS','tweet_count','heat_exposure','heat_norm']].head())

import numpy as np
cm = counties_merged
print(cm[['heat_exposure','heat_norm']].describe())
print('unique norm:', cm['heat_norm'].nunique())
print(cm.loc[cm.heat_norm.notna(),'STUSPS'].value_counts().loc[lambda s: s.index.isin(['AK','HI','PR'])])

import os, mapclassify, matplotlib.pyplot as plt
from matplotlib.patches import Patch
os.makedirs('pred_results', exist_ok=True)
vals_n = cm['heat_norm'].dropna().values
k = min(5, len(np.unique(vals_n)))
nb = mapclassify.NaturalBreaks(vals_n, k=k)
print(nb)
bins = nb.bins
cmap = plt.get_cmap('YlOrRd', k)
colors = [cmap(i) for i in range(k)]
def cls(v):
    return np.searchsorted(bins, v, side='left').clip(0, k-1)
g = cm.copy()
g['cls'] = np.where(g.heat_norm.notna(), cls(g.heat_norm.fillna(0).values), -1)
g['color'] = [colors[c] if c >= 0 else (0.85,0.85,0.85,1) for c in g['cls']]

def draw(ax, sub, crs):
    s = sub.to_crs(crs)
    s.plot(ax=ax, color=s['color'], edgecolor='white', linewidth=0.1)
    ax.set_axis_off()

fig = plt.figure(figsize=(16, 10))
ax = fig.add_axes([0.0, 0.08, 1.0, 0.85])
draw(ax, g[~g.STUSPS.isin(['AK','HI','PR'])], 'EPSG:5070')
states = g[~g.STUSPS.isin(['AK','HI','PR'])].to_crs('EPSG:5070').dissolve('STUSPS')
states.boundary.plot(ax=ax, color='#555555', linewidth=0.4)
ax_ak = fig.add_axes([0.02, 0.06, 0.22, 0.25]); draw(ax_ak, g[g.STUSPS=='AK'], 'EPSG:3338'); ax_ak.set_title('Alaska', fontsize=9)
ax_hi = fig.add_axes([0.25, 0.06, 0.15, 0.15]); draw(ax_hi, g[(g.STUSPS=='HI')], 'ESRI:102007'); ax_hi.set_title('Hawaii', fontsize=9)
ax_pr = fig.add_axes([0.42, 0.06, 0.10, 0.08]); draw(ax_pr, g[g.STUSPS=='PR'], 'EPSG:32161'); ax_pr.set_title('Puerto Rico', fontsize=9)
lo = [vals_n.min()] + list(bins[:-1])
labels = [f'{a:.3f} – {b:.3f} (n={c})' for a, b, c in zip(lo, bins, nb.counts)]
handles = [Patch(facecolor=colors[i], edgecolor='k', linewidth=0.3, label=labels[i]) for i in range(k)]
handles.append(Patch(facecolor=(0.85,0.85,0.85), edgecolor='k', linewidth=0.3, label=f'No weather tweets (n={int(g.heat_norm.isna().sum())})'))
ax.legend(handles=handles, title='Normalized heat exposure\n(Natural Breaks, k=%d)' % k, loc='lower right', fontsize=10, title_fontsize=11, frameon=True)
ax.set_title('Heat exposure sentiment from geotagged weather tweets by US county (2021-09-25)\nMean dictionary score per tweet, min-max normalized', fontsize=15)
out = 'pred_results/US_heat_exposure_map.png'
plt.savefig(out, dpi=200, bbox_inches='tight'); plt.close()
print(os.path.exists(out), os.path.getsize(out))

print('class match:', (cls(vals_n) == nb.yb).all(), np.bincount(g.loc[g.cls>=0,'cls']))
csv = cm.loc[cm.heat_norm.notna(), ['GEOID','NAME','STUSPS','heat_exposure','heat_norm']].merge(county_stats[['GEOID','tweet_count','score_sum']], on='GEOID').sort_values('heat_norm', ascending=False)
csv.to_csv('pred_results/county_heat_exposure.csv', index=False)
print(len(csv)); print(os.listdir('pred_results'))