# Wrong: Gi* is run on a 350 m square fishnet and only drawn as outlines over hexagon counts; the hexagons
# (clipped to the ZIP union) carry correct counts, but the map does not draw a hot spot result on them.
points_raw = gpd.read_file('dataset/High_Blood_Level_Results.shp')
print(f"Rows: {len(points_raw)}")
print(f"CRS: {points_raw.crs.to_string() if points_raw.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in points_raw.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in points_raw.total_bounds]}")
_cols = [c for c in points_raw.columns if c != points_raw.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(points_raw[_cols].head(3).to_string())


zips_raw = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp')
print(f"Rows: {len(zips_raw)}")
print(f"CRS: {zips_raw.crs.to_string() if zips_raw.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in zips_raw.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in zips_raw.total_bounds]}")
_cols = [c for c in zips_raw.columns if c != zips_raw.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(zips_raw[_cols].head(3).to_string())



import numpy as np
points_gdf = points_raw.to_crs("EPSG:32610").copy()
zips_gdf = zips_raw.to_crs("EPSG:32610").copy()

print("Points:", len(points_gdf), "| CRS:", points_gdf.crs.name, "| EPSG:", points_gdf.crs.to_epsg())
print("ZIPs:  ", len(zips_gdf), "| CRS:", zips_gdf.crs.name, "| EPSG:", zips_gdf.crs.to_epsg())
print("Points bounds:", np.round(points_gdf.total_bounds, 1))
print("ZIPs bounds:  ", np.round(zips_gdf.total_bounds, 1))

print("\nPoint empty:", points_gdf.geometry.is_empty.sum(), "| null:", points_gdf.geometry.isna().sum(),
      "| invalid:", (~points_gdf.geometry.is_valid).sum())
print("ZIP empty:", zips_gdf.geometry.is_empty.sum(), "| null:", zips_gdf.geometry.isna().sum(),
      "| invalid:", (~zips_gdf.geometry.is_valid).sum())

xy = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])


xy = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])
print("Duplicate point coordinates (rounded to mm):", len(xy) - len(np.unique(np.round(xy, 3), axis=0)))
print("Exact duplicate coordinates:", len(xy) - len(np.unique(xy, axis=0)))

geoms = list(zips_gdf.geometry)
ov = 0
for i in range(len(geoms)):
    for j in range(i+1, len(geoms)):
        if geoms[i].intersection(geoms[j]).area > 1:
            ov += 1
print("ZIP pairs with overlapping area > 1 m2:", ov)
zu = zips_gdf.union_all()
print("Points inside any ZIP polygon:", int(points_gdf.within(zu).sum()), "of", len(points_gdf))
print("Points intersecting ZIP union (incl. boundary):", int(points_gdf.intersects(zu).sum()))



import numpy as np, geopandas as gpd
from scipy.spatial import cKDTree
print(points_gdf.crs, len(points_gdf), points_gdf.geom_type.unique())
print(zips_gdf.crs, len(zips_gdf))
print(points_gdf.total_bounds, zips_gdf.total_bounds)
xy2 = np.column_stack([points_gdf.geometry.x, points_gdf.geometry.y])
tree = cKDTree(xy2)
d,_ = tree.query(xy2, k=2)
nn = d[:,1]
print("mean NN dist (m):", nn.mean(), "median:", np.median(nn), "min/max:", nn.min(), nn.max())
# Clark-Evans style: observed mean NN vs expected
area = zips_gdf.unary_union.area if hasattr(zips_gdf,'unary_union') else zips_gdf.geometry.union_all().area
print("study area km2:", area/1e6)



from shapely.geometry import box
from scipy.stats import norm
import shapely
cell = 350.0
study = zips_gdf.geometry.union_all()
minx,miny,maxx,maxy = zips_gdf.total_bounds
xs = np.arange(minx, maxx+cell, cell); ys = np.arange(miny, maxy+cell, cell)
polys=[]; cx=[]; cy=[]; ids=[]
for ix in range(len(xs)-1):
    for iy in range(len(ys)-1):
        b = box(xs[ix], ys[iy], xs[ix+1], ys[iy+1])
        if b.intersects(study):
            polys.append(b); cx.append((xs[ix]+xs[ix+1])/2); cy.append((ys[iy]+ys[iy+1])/2)
fish = gpd.GeoDataFrame({'cell_id': range(len(polys))}, geometry=polys, crs=points_gdf.crs)
# count points per cell
pj = gpd.sjoin(points_gdf[['geometry']], fish, how='left', predicate='within')
print("points matched to cells:", pj['cell_id'].notna().sum(), "of", len(points_gdf))
counts = pj.groupby('cell_id').size()
fish['count'] = fish['cell_id'].map(counts).fillna(0).astype(float)
print("cells:", len(fish), "cells with >=1 point:", (fish['count']>0).sum(), "max count:", fish['count'].max(), "total:", fish['count'].sum())



um = pj[pj['cell_id'].isna()]
print(um.geometry.iloc[0], "distance to study area (m):", um.geometry.iloc[0].distance(study))
# assign unmatched point to nearest cell (boundary case) so every case is counted
pts_un = points_gdf.loc[um.index]
nearest = gpd.sjoin_nearest(pts_un[['geometry']], fish[['cell_id','geometry']], how='left', distance_col='dist')
print(nearest[['cell_id','dist']])
fish.loc[fish['cell_id']==int(nearest['cell_id'].iloc[0]),'count'] += 1
print("total count now:", fish['count'].sum())

# Gi* on fishnet counts, binary distance band (includes self)
band = 3*cell   # 1050 m
cxy = np.column_stack([fish.geometry.centroid.x, fish.geometry.centroid.y])
t2 = cKDTree(cxy)
nbrs = t2.query_ball_point(cxy, r=band+1e-6)
n = len(fish); x = fish['count'].values
xbar = x.mean(); S = np.sqrt((x**2).sum()/n - xbar**2)
gi = np.empty(n); 
for i_,nb in enumerate(nbrs):
    w = np.zeros(0)
    wi = len(nb); sx = x[nb].sum()
    num = sx - xbar*wi
    den = S*np.sqrt((n*wi - wi**2)/(n-1))
    gi[i_] = num/den if den>0 else np.nan
fish['gi_z'] = gi
fish['nbr_n'] = [len(nb) for nb in nbrs]
fish['p_value'] = 2*norm.sf(np.abs(fish['gi_z']))
print("neighbours per cell: min", fish.nbr_n.min(), "median", fish.nbr_n.median(), "max", fish.nbr_n.max())
print("gi_z range:", np.nanmin(gi), np.nanmax(gi), "NaN:", np.isnan(gi).sum())



def classify(z):
    if z >= 2.576: return 'Hot 99%'
    if z >= 1.96:  return 'Hot 95%'
    if z >= 1.645: return 'Hot 90%'
    if z <= -2.576: return 'Cold 99%'
    if z <= -1.96:  return 'Cold 95%'
    if z <= -1.645: return 'Cold 90%'
    return 'Not significant'
fish['conf_class'] = [classify(z) for z in fish['gi_z']]
fish['hot_sig_90'] = fish['gi_z']>=1.645
fish['hot_sig_95'] = fish['gi_z']>=1.96
fish['hot_sig_99'] = fish['gi_z']>=2.576
print(fish['conf_class'].value_counts())
print("hot cells 90/95/99:", fish.hot_sig_90.sum(), fish.hot_sig_95.sum(), fish.hot_sig_99.sum())
print("cells with count>=1 among hot99:", (fish.hot_sig_99 & (fish['count']>0)).sum())
print("mean count in hot99 cells:", fish.loc[fish.hot_sig_99,'count'].mean(), "overall mean:", fish['count'].mean())
gi_grid = fish.copy()
gi_grid['cell_size_m'] = cell; gi_grid['band_m'] = band
print(gi_grid.head(3)[['cell_id','count','gi_z','p_value','conf_class']])


print(gi_grid.columns.tolist())
print(gi_grid['conf_class'].value_counts(dropna=False))
print(gi_grid[['count','gi_z','p_value']].describe())
print(zips_gdf.total_bounds, zips_gdf.crs)
print(pj.shape, points_gdf.shape)


import numpy as np, geopandas as gpd, math
from shapely.geometry import Polygon
from shapely.ops import unary_union
print(gi_grid['hot_sig_90'].sum(), gi_grid['hot_sig_95'].sum(), gi_grid['hot_sig_99'].sum())
HEX_SIDE = 1000.0
study_union = unary_union(zips_gdf.geometry.buffer(0))
minx_, miny_, maxx_, maxy_ = study_union.bounds
dx = 1.5*HEX_SIDE; dy = math.sqrt(3)*HEX_SIDE
hexes = []
col = 0
x0 = minx_ - 2*HEX_SIDE
while x0 <= maxx_ + 2*HEX_SIDE:
    yoff = (dy/2) if col % 2 else 0.0
    y0 = miny_ - 2*HEX_SIDE + yoff
    while y0 <= maxy_ + 2*HEX_SIDE:
        pts = [(x0 + HEX_SIDE*math.cos(math.radians(a)), y0 + HEX_SIDE*math.sin(math.radians(a))) for a in range(0, 360, 60)]
        hexes.append(Polygon(pts))
        y0 += dy
    x0 += dx; col += 1
hex_all = gpd.GeoDataFrame({'hex_id': range(len(hexes))}, geometry=hexes, crs=zips_gdf.crs)
# keep hexes that overlap the study area (ZIP union)
hex_all = hex_all[hex_all.intersects(study_union)].copy()
hex_all['hex_id'] = range(len(hex_all))
hex_all['geometry'] = hex_all.geometry.intersection(study_union)  # clip to study area boundary
hex_all = hex_all[~hex_all.geometry.is_empty].reset_index(drop=True)
hex_all['hex_id'] = range(len(hex_all))
# point-in-polygon counts
pts_in = gpd.sjoin(points_gdf[['geometry']], hex_all[['hex_id','geometry']], how='left', predicate='within')
print('points matched to hexes:', pts_in['hex_id'].notna().sum(), 'of', len(points_gdf))
cnt = pts_in.groupby('hex_id').size()
hex_all['case_count'] = hex_all['hex_id'].map(cnt).fillna(0).astype(int)
hex_all['suppressed'] = hex_all['case_count'] < 5
# hot-spot overlap with significant (90%) hot cells
hot_cells = gi_grid[gi_grid['hot_sig_90'].astype(bool)]
hot_union = unary_union(hot_cells.geometry)
hex_all['hot_area_frac'] = hex_all.geometry.apply(lambda g: g.intersection(hot_union).area / g.area if g.area>0 else 0)
hex_all['hot_spot'] = hex_all['hot_area_frac'] > 0
print('hexagons:', len(hex_all), 'suppressed (<5):', hex_all['suppressed'].sum(), 'hot_spot flagged:', hex_all['hot_spot'].sum())
print('case_count range:', hex_all.case_count.min(), hex_all.case_count.max(), 'sum:', hex_all.case_count.sum())
print(hex_all.sort_values('case_count', ascending=False)[['hex_id','case_count','suppressed','hot_spot','hot_area_frac']].head(10))


unmatched = pts_in[pts_in['hex_id'].isna()]
print(len(unmatched), unmatched.geometry.tolist())
print('outside study union:', (~points_gdf.geometry.within(study_union)).sum(), 'touching study union:', points_gdf.geometry.intersects(study_union).sum())
print('dist to study union (m) for unmatched:', [round(study_union.exterior.distance(g) if False else study_union.distance(g),2) for g in unmatched.geometry])
print('hot_area_frac distribution (flagged):', hex_all.loc[hex_all.hot_spot,'hot_area_frac'].describe())


HOT_MIN_FRAC = 0.10
hex_gdf = hex_all.copy()
hex_gdf['hot_spot'] = hex_gdf['hot_area_frac'] >= HOT_MIN_FRAC
print('hexagons:', len(hex_gdf))
print('suppressed (<5 cases):', int(hex_gdf.suppressed.sum()), '| not suppressed:', int((~hex_gdf.suppressed).sum()))
print('hot_spot flagged (overlap>=10%):', int(hex_gdf.hot_spot.sum()), '| any-overlap>0:', int((hex_gdf.hot_area_frac>0).sum()))
print('hot_spot & suppressed:', int((hex_gdf.hot_spot & hex_gdf.suppressed).sum()))
print('cases in non-suppressed hexes:', int(hex_gdf.loc[~hex_gdf.suppressed,'case_count'].sum()), 'of', len(points_gdf), '(3 points lie outside ZIP study area, unassigned)')
print(hex_gdf.case_count.describe())


print(gi_grid.columns.tolist()); print(gi_grid['conf_class'].value_counts(dropna=False))
print(hex_gdf.columns.tolist()); print(hex_gdf[['case_count','suppressed','hot_spot']].describe(include='all'))
print(points_gdf.crs, len(points_gdf), points_gdf.geometry.isna().sum())
print(hex_gdf.crs, len(hex_gdf))

import geopandas as gpd, pandas as pd
hot_cells = gi_grid[gi_grid['hot_sig_95'].astype(bool)][['cell_id','conf_class','geometry']]
hot_hex = hex_gdf[hex_gdf['hot_spot'].astype(bool)][['hex_id','geometry']]
print('hot cells (95%):', len(hot_cells), ' hot hexes:', len(hot_hex))
pts = points_gdf[['geometry']].copy()
pts['pt_id'] = range(1, len(pts)+1)
j_cell = gpd.sjoin(pts, hot_cells, how='inner', predicate='within')
j_hex = gpd.sjoin(pts, hot_hex, how='inner', predicate='within')
in_cell_ids = set(j_cell['pt_id']); in_hex_ids = set(j_hex['pt_id'])
print('points in hot cells:', len(in_cell_ids), ' in hot hexes:', len(in_hex_ids), ' union:', len(in_cell_ids|in_hex_ids))
sel = pts[pts['pt_id'].isin(in_cell_ids|in_hex_ids)].copy()
sel['in_hot_cell'] = sel['pt_id'].isin(in_cell_ids).astype(int)
sel['in_hot_hex'] = sel['pt_id'].isin(in_hex_ids).astype(int)
# conf class of containing cell (highest-confidence match)
cc = j_cell.sort_values('conf_class').groupby('pt_id')['conf_class'].first()
sel['hot_class'] = sel['pt_id'].map(cc).fillna('Hot (hex only)')
hotspot_points = sel.reset_index(drop=True)
print(len(hotspot_points), 'of', len(points_gdf), f'= {len(hotspot_points)/len(points_gdf):.1%} of cases')
print(hotspot_points['hot_class'].value_counts())
print(hotspot_points.head(3))

order = {'Hot 99%':0,'Hot 95%':1,'Hot 90%':2}
jc = j_cell.copy(); jc['o'] = jc['conf_class'].map(order)
cc2 = jc.sort_values('o').groupby('pt_id')['conf_class'].first()
hotspot_points = pts[pts['pt_id'].isin(in_cell_ids|in_hex_ids)].copy().reset_index(drop=True)
hotspot_points['hot_class'] = hotspot_points['pt_id'].map(cc2).fillna('Hot (flagged hex only)')
hotspot_points = hotspot_points[['pt_id','hot_class','geometry']]
print(hotspot_points['hot_class'].value_counts())
print(len(hotspot_points), 'points =', f'{len(hotspot_points)/len(points_gdf):.1%} of {len(points_gdf)} cases', hotspot_points.crs, hotspot_points.geom_type.unique())


import os
os.makedirs('pred_results', exist_ok=True)
for name in ['hex_gdf','gi_grid','hotspot_points','zips_gdf']:
    g = globals()[name]
    print(name, type(g).__name__, len(g), g.crs.to_string(), list(g.columns), g.geometry.geom_type.value_counts().to_dict())
print(hex_gdf[['case_count','suppressed','hot_spot']].describe(include='all'))
print(gi_grid['conf_class'].value_counts())
print(hotspot_points.head())



import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

hx = hex_gdf.copy()
print('case_count dtype', hx['case_count'].dtype, 'min non-suppressed', hx.loc[~hx.suppressed,'case_count'].min(), 'max', hx['case_count'].max())
print('suppressed with count>=5:', int((hx.suppressed & (hx.case_count>=5)).sum()), '| non-suppressed with count<5:', int((~hx.suppressed & (hx.case_count<5)).sum()))

hot95 = gi_grid[gi_grid.hot_sig_95.astype(bool)].dissolve().geometry.iloc[0] if gi_grid.hot_sig_95.astype(bool).any() else None
hot99 = gi_grid[gi_grid.hot_sig_99.astype(bool)].dissolve().geometry.iloc[0]
hot90 = gi_grid[gi_grid.hot_sig_90.astype(bool)].dissolve().geometry.iloc[0]

fig, ax = plt.subplots(figsize=(10, 10))
# unsuppressed hexagons shaded by count
ok = hx[~hx.suppressed]
sup = hx[hx.suppressed]
ok.plot(column='case_count', cmap='YlOrRd', vmin=5, vmax=hx.case_count.max(), edgecolor='grey', linewidth=0.3, ax=ax, legend=False)
# suppressed hexagons hatched
sup.plot(facecolor='none', edgecolor='dimgrey', hatch='///', linewidth=0.3, ax=ax)
# ZIP boundaries
zips_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
# hot spot outlines
gpd_hot = gpd = None
import geopandas as gpd
for geom, ls, col, lab in [(hot90,':','#1f78b4','Hot spot 90%'),(hot90 if False else hot95,'--','#1f78b4','Hot spot 95%'),(hot99,'-','#08306b','Hot spot 99%')]:
    if geom is not None and not geom.is_empty:
        gpd.GeoSeries([geom], crs=32610).boundary.plot(ax=ax, color=col, linestyle=ls, linewidth=1.8)
# labels for ZIP codes not needed; axis, legend
sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(5, hx.case_count.max()))
cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
cb.set_label('Elevated blood lead cases per hexagon (≥5 shown)')
handles = [Patch(facecolor='none', edgecolor='dimgrey', hatch='///', label='Suppressed (<5 cases)'),
           Line2D([0],[0], color='black', lw=0.8, label='ZIP code boundary'),
           Line2D([0],[0], color='#1f78b4', lw=1.8, ls=':', label='Hot spot 90% outline'),
           Line2D([0],[0], color='#1f78b4', lw=1.8, ls='--', label='Hot spot 95% outline'),
           Line2D([0],[0], color='#08306b', lw=1.8, ls='-', label='Hot spot 99% outline')]
ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.9)
ax.set_title('Elevated blood lead levels in children, Sacramento\nHexagon summary and Gi* hot spot areas (individual cases not shown)', fontsize=12)
ax.set_axis_off()
plt.savefig('pred_results/blood_lead_hotspots.png', dpi=200, bbox_inches='tight')
plt.close()
print('saved png')



hs = hex_gdf.copy()
hs['case_count_released'] = hs['case_count'].where(~hs['suppressed'].astype(bool))
hs = hs.drop(columns=['case_count'])
hs_out = hs.to_crs(4326)
hs_out.to_file('pred_results/hex_summary.geojson', driver='GeoJSON')
hotspot_points[['pt_id','hot_class','geometry']].to_file('pred_results/hotspot_points.shp')
chk_hex = gpd.read_file('pred_results/hex_summary.geojson')
chk_pts = gpd.read_file('pred_results/hotspot_points.shp')
print('hex_summary:', len(chk_hex), chk_hex.columns.tolist(), 'CRS', chk_hex.crs.to_string(), 'released non-null', chk_hex.case_count_released.notna().sum())
print('hotspot_points:', len(chk_pts), chk_pts.columns.tolist(), chk_pts.crs.to_string(), 'share of 373 =', round(len(chk_pts)/373*100,1))
print('ranges: hot_class', chk_pts.hot_class.value_counts().to_dict())
print(os.listdir('pred_results'))
