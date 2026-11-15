"""Correct: scores the full tweet text and skips place boxes of state / country size; lower-cased
tokens in the cased dictionary.
"""
import json, re, pandas as pd, geopandas as gpd, matplotlib.pyplot as plt
import mapclassify
from shapely.geometry import Point
scores = {}
for item in open('dataset/data20000.txt').read().strip().strip('{}').split(', '):
    k, _, v = item.rpartition(':')
    try:
        scores[k.strip().strip("'\"")] = float(v)
    except ValueError:
        pass
tweets = json.load(open('dataset/12500-tweets-2021-09-24_01-48-23.json'))
recs = []
for t in tweets:
    text = (t.get('extended_tweet') or {}).get('full_text') or t.get('text') or ''
    s = sum(scores.get(w, 0) for w in re.findall(r'\w+', text.lower()))
    if s == 0:
        continue
    if t.get('coordinates'):
        lon, lat = t['coordinates']['coordinates']
    elif t.get('place') and t['place']['place_type'] not in ('admin', 'country'):   # skip state/country boxes
        bb = t['place']['bounding_box']['coordinates'][0]
        lon = sum(p[0] for p in bb) / 4; lat = sum(p[1] for p in bb) / 4
    else:
        continue
    recs.append({'score': s, 'geometry': Point(lon, lat)})
pts = gpd.GeoDataFrame(recs, crs=4326)
counties = gpd.read_file('dataset/cb_2020_us_county_20m.shp').to_crs(4326)
j = gpd.sjoin(pts, counties[['GEOID', 'geometry']], predicate='within')
agg = j.groupby('GEOID').score.agg(['count', 'mean']).reset_index()
agg['heat_norm'] = (agg['mean'] - agg['mean'].min()) / (agg['mean'].max() - agg['mean'].min())
m = counties.merge(agg, on='GEOID', how='left')
m = m[~m.STATEFP.isin(['02', '15', '72'])]
ax = m.plot(color='lightgrey', figsize=(16, 9))
m.dropna(subset=['heat_norm']).plot(column='heat_norm', scheme='FisherJenks', k=5, cmap='Reds', legend=True, ax=ax)
plt.savefig('pred_results/US_heat_exposure_map.png')
