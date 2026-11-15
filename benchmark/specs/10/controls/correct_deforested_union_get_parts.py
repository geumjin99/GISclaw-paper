# Correct: the deforested area is drawn after shapely.union_all + get_parts. The steps run in order; steps
# that may raise are wrapped in try/except.
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


roads = gpd.read_file('dataset/roads.geojson')
print(f"Rows: {len(roads)}")
print(f"CRS: {roads.crs.to_string() if roads.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in roads.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in roads.total_bounds]}")
_cols = [c for c in roads.columns if c != roads.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(roads[_cols].head(3).to_string())

deforested = gpd.read_file('dataset/deforestedArea.geojson')
print(f"Rows: {len(deforested)}")
print(f"CRS: {deforested.crs.to_string() if deforested.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in deforested.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in deforested.total_bounds]}")
_cols = [c for c in deforested.columns if c != deforested.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(deforested[_cols].head(3).to_string())

protected = gpd.read_file('dataset/protectedForest.geojson')
print(f"Rows: {len(protected)}")
print(f"CRS: {protected.crs.to_string() if protected.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in protected.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in protected.total_bounds]}")
_cols = [c for c in protected.columns if c != protected.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(protected[_cols].head(3).to_string())

planned = gpd.read_file('dataset/planned_road.geojson')
print(f"Rows: {len(planned)}")
print(f"CRS: {planned.crs.to_string() if planned.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in planned.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in planned.total_bounds]}")
_cols = [c for c in planned.columns if c != planned.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(planned[_cols].head(3).to_string())

import geopandas as gpd
import numpy as np

planned_4326 = planned.to_crs(4326)
print(planned_4326.total_bounds)
print(planned_4326['Name'], planned_4326['Status'])
# length in projected CRS (meters)
print("Planned road length (km):", planned.geometry.length[0]/1000)

# check state extents from the layers
print("roads extent:", roads.total_bounds)
print("deforested extent:", deforested.total_bounds)
print("protected extent:", protected.total_bounds)
print("planned (4326) extent:", planned_4326.total_bounds)

import subprocess, os
# search for potential brazil state boundary files on the system
res = subprocess.run(["bash","-lc","find / -iname '*rond*' 2>/dev/null | head -20; find / -iname '*states*' -path '*shp*' 2>/dev/null | head -20; find / -iname 'br_states*' 2>/dev/null | head; find / -iname '*admin*' -iname '*.geojson' 2>/dev/null | head"], capture_output=True, text=True)
print(res.stdout)
print(res.stderr[:500])

try:  # errors in this block are tolerated
    import geopandas as gpd
    ne = gpd.read_file("/opt/cartopy/shapefiles/natural_earth/cultural/ne_10m_admin_1_states_provinces_lakes.shp")
    print(ne.columns.tolist())
    br = ne[ne['admin']=='Brazil'].copy()
    print(br[['name','admin','shapeName']].head(30))
    ron = br[br['name'].str.contains('Rond', case=False, na=False)]
    print(ron[['name','admin','id_0']].to_string())
    print(ron.crs, ron.total_bounds)
except Exception:
    pass

import geopandas as gpd
br = ne[ne['admin']=='Brazil']
print(br[['name','region','sov_a3']].to_string())
ron = br[br['name'].str.contains('Rond', case=False, na=False)].copy()
print(ron[['name','region']].to_string())
print(ron.crs, "bounds:", ron.total_bounds, "area km2:", ron.area.sum())

try:  # errors in this block are tolerated
    import geopandas as gpd
    import numpy as np

    P = 31919
    roads_p   = roads.to_crs(P)
    defo_p    = deforested.to_crs(P)
    prot_p    = protected.to_crs(P)
    plan_p    = planned.to_crs(P)
    ron_p     = ron.to_crs(P)

    # 5.5 km buffer around planned road
    plan_buf = gpd.GeoDataFrame(geometry=[plan_p.geometry[0].buffer(5500)], crs=P)
    print("buffer area km2:", plan_buf.area[0]/1e6)

    # risk area = buffer - deforested - protected, clipped to state
    base = plan_buf
    step1 = base.difference(defo_p)
    step2 = step1.difference(prot_p)
    risk  = step2.intersection(ron_p)
    risk = risk[~risk.geometry.is_empty]
    print("risk area km2:", risk.area.sum()/1e6)
    print("risk n parts:", len(risk))
    print("total buffer km2:", plan_buf.area[0]/1e6)
except Exception:
    pass

import pyproj
from pyproj import CRS
# UTM zone 19S, WGS84
utm19s = CRS.from_proj4("+proj=utm +zone=19 +south +datum=WGS84 +units=m +no_defs")
print(utm19s)
P = "+proj=utm +zone=19 +south +datum=WGS84 +units=m +no_defs"
roads_p   = roads.to_crs(P)
defo_p    = deforested.to_crs(P)
prot_p    = protected.to_crs(P)
plan_p    = planned.to_crs(P)
ron_p     = ron.to_crs(P)
print("planned bounds:", plan_p.total_bounds)
print("state bounds:", ron_p.total_bounds)

try:  # errors in this block are tolerated
    import geopandas as gpd

    P = "+proj=utm +zone=19 +south +datum=WGS84 +units=m +no_defs"
    plan_buf = gpd.GeoDataFrame(geometry=[plan_p.geometry[0].buffer(5500, resolution=32)], crs=P)
    print("buffer area km2:", plan_buf.area[0]/1e6)

    step1 = plan_buf.difference(defo_p)          # remove already deforested
    step2 = step1.difference(prot_p)              # remove protected forest
    risk  = step2.intersection(ron_p)             # clip to state boundary
    risk = risk[~risk.geometry.is_empty]
    print("risk area km2:", risk.area.sum()/1e6)
    print("risk parts:", len(risk))

    # breakdown: buffer portions already deforested / inside protected
    defo_part = plan_buf.intersection(defo_p).area.sum()/1e6
    prot_part = plan_buf.intersection(prot_p).area.sum()/1e6
    print("buffer already deforested km2:", defo_part, "| buffer in protected km2:", prot_part)
except Exception:
    pass

import geopandas as gpd
import shapely

P = "+proj=utm +zone=19 +south +datum=WGS84 +units=m +no_defs"

# validate all layers
def fix(gdf):
    g = gdf.copy()
    g['geometry'] = shapely.make_valid(g.geometry)
    return g

defo_p  = fix(defo_p)
prot_p  = fix(prot_p)
plan_p  = fix(plan_p)
ron_p   = fix(ron_p)
roads_p = fix(roads_p)

print("validity before fix check:")
print("defo valid:", all(shapely.is_valid(defo_p.geometry)), 
      "| prot valid:", all(shapely.is_valid(prot_p.geometry)),
      "| plan valid:", all(shapely.is_valid(plan_p.geometry)))
print("defo valid:", all(shapely.is_valid(defo_p.geometry)),
      "| prot valid:", all(shapely.is_valid(prot_p.geometry)),
      "| plan valid:", all(shapely.is_valid(plan_p.geometry)))

import geopandas as gpd
import shapely

P = "+proj=utm +zone=19 +south +datum=WGS84 +units=m +no_defs"
plan_buf = gpd.GeoDataFrame(geometry=[shapely.make_valid(plan_p.geometry[0]).buffer(5500)], crs=P)
print("buffer area km2:", plan_buf.area[0]/1e6)

step1 = plan_buf.difference(defo_p)
step2 = step1.difference(prot_p)
risk  = step2.intersection(ron_p)
risk = risk[~risk.geometry.is_empty].copy()
print("risk area km2:", risk.area.sum()/1e6, "| parts:", len(risk))

defo_part = plan_buf.intersection(defo_p).area.sum()/1e6
prot_part = plan_buf.intersection(prot_p).area.sum()/1e6
print("buffer already deforested km2:", round(defo_part,1), "| buffer in protected km2:", round(prot_part,1))

import shapely
print("plan_buf area:", plan_buf.geometry.area.sum()/1e6, "km2")
print("defo total area:", defo_p.geometry.area.sum()/1e6, "km2")
print("step1 area:", step1.area.sum()/1e6, "km2")
print("step2 area:", step2.area.sum()/1e6, "km2")
print("risk n:", len(risk))
print("risk area raw sum:", risk.area.sum())
print("risk area first 5:", risk.area[:5].tolist())
# check whether intersection produced slivers; also try without clipping to state
risk2 = step2  # no state clip
print("step2 (no clip) area:", risk2.area.sum()/1e6, "km2")

try:  # errors in this block are tolerated
    import shapely
    g = ron_p.geometry[0]
    print("state valid:", shapely.is_valid(g))
    print("state area:", shapely.area(g)/1e6, "km2")
    print("state geom type:", g.geom_type)
    # try intersection with a valid state
    gv = shapely.make_valid(g)
    print("fixed state valid:", shapely.is_valid(gv), "area:", shapely.area(gv)/1e6)
    r3 = shapely.intersection(step2.union_all(), gv)
    print("r3 area:", shapely.area(r3)/1e6, "km2")
    print("r3 valid:", shapely.is_valid(r3))
except Exception:
    pass

import shapely
g = ron_p.geometry.iloc[0]
print("state valid:", shapely.is_valid(g))
print("state area:", shapely.area(g)/1e6, "km2")
gv = shapely.make_valid(g)
print("fixed state valid:", shapely.is_valid(gv), "area:", shapely.area(gv)/1e6)
r3 = shapely.intersection(shapely.union_all(step2.to_list()), gv)
print("r3 area:", shapely.area(r3)/1e6, "km2", "| valid:", shapely.is_valid(r3))

try:  # errors in this block are tolerated
    import geopandas as gpd
    import shapely

    P = "+proj=utm +zone=19 +south +datum=WGS84 +units=m +no_defs"
    buf = shapely.make_valid(plan_p.geometry.iloc[0]).buffer(5500)
    buf_gdf = gpd.GeoDataFrame(geometry=[buf], crs=P)

    s1 = buf_gdf.difference(defo_p, align=False).geometry.iloc[0]
    s2 = s1.difference(shapely.union_all(prot_p.geometry), align=False) if hasattr(s1,'difference') else s1
    # careful: s1 is a shapely geom
    s2 = shapely.difference(s1, shapely.union_all(prot_p.geometry))
    risk_geom = shapely.intersection(s2, shapely.union_all(ron_p.geometry))
    risk_geom = shapely.make_valid(risk_geom)
    risk_gdf = gpd.GeoDataFrame(geometry=shapely.get_parts(risk_geom) if risk_geom.geom_type=='MultiPolygon' else [risk_geom], crs=P)

    area = lambda x: shapely.area(x)/1e6
    print(f"buffer total:            {area(buf):.1f} km2")
    print(f"buffer in already defo:  {area(shapely.intersection(buf, shapely.union_all(defo_p.geometry))):.1f} km2")
    print(f"buffer in protected:     {area(shapely.intersection(buf, shapely.union_all(prot_p.geometry))):.1f} km2")
    print(f"buffer outside state:    {area(shapely.difference(buf, shapely.union_all(ron_p.geometry))):.1f} km2")
    print(f"AT-RISK area:            {area(risk_geom):.1f} km2  ({100*area(risk_geom)/area(buf):.1f}% of buffer)")
    print("risk parts:", len(risk_gdf))
except Exception:
    pass

import geopandas as gpd
import shapely

P = "+proj=utm +zone=19 +south +datum=WGS84 +units=m +no_defs"
area = lambda x: shapely.area(x)/1e6

buf = shapely.make_valid(plan_p.geometry.iloc[0]).buffer(5500)
defo_all = shapely.union_all(defo_p.geometry)
prot_all = shapely.union_all(prot_p.geometry)
state_all = shapely.union_all(ron_p.geometry)

s1 = shapely.difference(buf, defo_all)
s2 = shapely.difference(s1, prot_all)
risk_geom = shapely.make_valid(shapely.intersection(s2, state_all))

print(f"buffer total:            {area(buf):.1f} km2")
print(f"buffer in already defo:  {area(shapely.intersection(buf, defo_all)):.1f} km2")
print(f"buffer in protected:     {area(shapely.intersection(buf, prot_all)):.1f} km2")
print(f"buffer outside state:    {area(shapely.difference(buf, state_all)):.1f} km2")
print(f"AT-RISK area:            {area(risk_geom):.1f} km2  ({100*area(risk_geom)/area(buf):.1f}% of buffer)")

parts = shapely.get_parts(risk_geom) if risk_geom.geom_type in ('MultiPolygon','GeometryCollection') else shapely.geometry.collect([risk_geom])
risk_gdf = gpd.GeoDataFrame(geometry=list(parts), crs=P)
risk_gdf = risk_gdf[~risk_gdf.geometry.is_empty]
print("risk parts:", len(risk_gdf), "| area check:", risk_gdf.area.sum()/1e6, "km2")

try:  # errors in this block are tolerated
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    import geopandas as gpd
    import shapely

    # geometry prep (all in UTM 19S)
    state = ron_p.geometry.iloc[0]
    buf = shapely.make_valid(plan_p.geometry.iloc[0]).buffer(5500)
    risk = shapely.make_valid(shapely.intersection(shapely.difference(
        shapely.difference(buf, shapely.union_all(defo_p.geometry)),
        shapely.union_all(prot_p.geometry)), state))
    plan_line = plan_p.geometry.iloc[0]

    def draw_map(ax, xmin, ymin, xmax, ymax, show_all_state=True):
        # state boundary / backdrop
        if show_all_state:
            ax.add_patch(plt.Polygon(state.exterior.coords, closed=True,
                         facecolor='#F7F3E8', edgecolor='#333333', linewidth=2.2, zorder=1))
        # deforested
        for g in shapely.get_parts(defo_all) if defo_all.geom_type != 'Polygon' else [defo_all]:
            pass
        dg = gpd.GeoDataFrame(geometry=shapely.get_parts(defo_all) if defo_all.geom_type != 'Polygon' else [defo_all], crs=P)
        dg.plot(ax=ax, color='#F2B58C', edgecolor='none', alpha=0.85, zorder=2)
        # protected forest
        prot_p.plot(ax=ax, color='#B5D99A', edgecolor='#7FAE5D', linewidth=0.4, alpha=0.9, zorder=3)
        # existing roads
        roads_p.plot(ax=ax, color='#555555', linewidth=0.35, zorder=4)
        # buffer outline
        bx, by = buf.exterior.xy
        ax.plot(bx, by, color='#2C6FBB', linewidth=1.6, linestyle=':', zorder=5,
                label='5.5 km buffer around planned road')
        # at-risk area
        rp = gpd.GeoDataFrame(geometry=list(shapely.get_parts(risk)), crs=P)
        rp.plot(ax=ax, color='#D7301F', edgecolor='#8B0000', linewidth=0.3, alpha=0.85, zorder=6,
                label='At risk of new deforestation (566.5 km²)')
        # planned road
        for g in shapely.get_parts(plan_line):
            xx, yy = g.xy
            ax.plot(xx, yy, color='#FFD400', linewidth=3.2, zorder=7,
                    path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal'); ax.axis('off')

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.05)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    sb = state.bounds
    draw_map(ax1, sb.minx-15000, sb.miny-15000, sb.maxx+15000, sb.maxy+15000)
    ax1.set_title('Rondônia – Deforestation, protected areas & existing roads', fontsize=13, pad=8)
    ax1.text(0.01, 0.02, 'State boundary: Natural Earth 10m admin-1', transform=ax1.transAxes, fontsize=8, color='#555')

    bb = buf.bounds
    m = 30000
    draw_map(ax2, bb.minx-m, bb.miny-m, bb.maxx+m, bb.maxy+m, show_all_state=False)
    ax2.set_title('Zoom: planned BR-421 (208 km) & 5.5 km risk buffer', fontsize=13, pad=8)
    ax2.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

    # legend for left map
    handles, labels = ax2.get_legend_handles_labels()
    ax1.legend(handles=handles, loc='lower left', fontsize=8.5, framealpha=0.9)

    # extra legend for planned road on ax1
    xx, yy = shapely.get_parts(plan_line)[0].xy
    ax1.plot(xx, yy, color='#FFD400', linewidth=3.0, zorder=7,
             label='Planned road (Proposed BR-421)',
             path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
    ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

    fig.suptitle('Effect of the planned BR-421 road on deforestation in Rondônia, Brazil\n'
                 '5.5 km buffer: 2,383.5 km² total | at risk of new deforestation: 566.5 km² (23.8%)',
                 fontsize=15, fontweight='bold', y=0.985)

    # scale bar on ax1
    def scale_bar(ax, x, y, length_m=100000):
        ax.plot([x, x+length_m], [y, y], color='k', linewidth=3)
        for xx in (x, x+length_m/2, x+length_m):
            ax.plot([xx, xx], [y-2000, y+2000], color='k', linewidth=1.5)
        ax.text(x+length_m/2, y+6000, '100 km', ha='center', fontsize=9)
    scale_bar(ax1, sb.minx+30000, sb.miny+40000)
    scale_bar(ax2, bb.minx+10000, bb.miny+12000, 20000)
    ax2.text(bb.minx+20000, bb.miny+40000, '20 km', ha='left', fontsize=8)

    import os
    os.makedirs('pred_results', exist_ok=True)
    fig.savefig('pred_results/predictedRiskyArea.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("saved")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import geopandas as gpd
import shapely, os

P = "+proj=utm +zone=19 +south +datum=WGS84 +units=m +no_defs"
state = ron_p.geometry.iloc[0]
buf = shapely.make_valid(plan_p.geometry.iloc[0]).buffer(5500)
risk = shapely.make_valid(shapely.intersection(shapely.difference(
    shapely.difference(buf, shapely.union_all(defo_p.geometry)),
    shapely.union_all(prot_p.geometry)), state))
plan_line = plan_p.geometry.iloc[0]

defo_parts = list(shapely.get_parts(defo_all)) if defo_all.geom_type != 'Polygon' else [defo_all]
risk_parts = list(shapely.get_parts(risk)) if risk.geom_type in ('MultiPolygon','GeometryCollection') else [risk]
dg = gpd.GeoDataFrame(geometry=defo_parts, crs=P)
rp = gpd.GeoDataFrame(geometry=risk_parts, crs=P)
pln_parts = list(shapely.get_parts(plan_line))

def draw_map(ax, xmin, ymin, xmax, ymax, show_state=True):
    if show_state:
        ax.add_patch(plt.Polygon(state.exterior.coords, closed=True,
                     facecolor='#F7F3E8', edgecolor='#333333', linewidth=2.2, zorder=1))
    dg.plot(ax=ax, color='#F2B58C', edgecolor='none', alpha=0.85, zorder=2)
    prot_p.plot(ax=ax, color='#B5D99A', edgecolor='#7FAE5D', linewidth=0.4, alpha=0.9, zorder=3)
    roads_p.plot(ax=ax, color='#555555', linewidth=0.35, zorder=4)
    bx, by = buf.exterior.xy
    ax.plot(bx, by, color='#2C6FBB', linewidth=1.6, linestyle=':', zorder=5,
            label='5.5 km buffer around planned road')
    rp.plot(ax=ax, color='#D7301F', edgecolor='#8B0000', linewidth=0.3, alpha=0.85, zorder=6,
            label='At risk of new deforestation (566.5 km²)')
    for g in pln_parts:
        xx, yy = g.xy
        ax.plot(xx, yy, color='#FFD400', linewidth=3.2, zorder=7,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal'); ax.axis('off')

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.05)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

sx0, sy0, sx1, sy1 = state.bounds
draw_map(ax1, sx0-15000, sy0-15000, sx1+15000, sy1+15000)
ax1.set_title('Rondônia – deforested area, protected forest & existing roads', fontsize=13, pad=8)
ax1.text(0.01, 0.02, 'State boundary: Natural Earth 10m admin-1', transform=ax1.transAxes, fontsize=8, color='#555')

bx0, by0, bx1, by1 = buf.bounds
m = 30000
draw_map(ax2, bx0-m, by0-m, bx1+m, by1+m, show_state=False)
ax2.set_title('Zoom: planned BR-421 (208 km) & 5.5 km risk buffer', fontsize=13, pad=8)
ax2.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

# planned road legend entry on left map
xx, yy = pln_parts[0].xy
ax1.plot(xx, yy, color='#FFD400', linewidth=3.0, zorder=7,
         label='Planned road (Proposed BR-421)',
         path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
ax1.legend(loc='lower left', fontsize=8.5, framealpha=0.9)

fig.suptitle('Effect of the planned BR-421 road on deforestation in Rondônia, Brazil\n'
             '5.5 km buffer: 2,383.5 km² total  |  at risk of new deforestation: 566.5 km² (23.8% of buffer)',
             fontsize=15, fontweight='bold', y=0.985)

def scale_bar(ax, x, y, length_m, label):
    ax.plot([x, x+length_m], [y, y], color='k', linewidth=3)
    for xx in (x, x+length_m/2, x+length_m):
        ax.plot([xx, xx], [y-2000, y+2000], color='k', linewidth=1.5)
    ax.text(x+length_m/2, y+8000, label, ha='center', fontsize=9)
scale_bar(ax1, sx0+40000, sy0+50000, 100000, '100 km')
scale_bar(ax2, bx0+12000, by0+14000, 20000, '20 km')

os.makedirs('pred_results', exist_ok=True)
fig.savefig('pred_results/predictedRiskyArea.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("saved figure")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
import geopandas as gpd
import shapely, os

state = ron_p.geometry.iloc[0]
buf = shapely.make_valid(plan_p.geometry.iloc[0]).buffer(5500)
risk = shapely.make_valid(shapely.intersection(shapely.difference(
    shapely.difference(buf, shapely.union_all(defo_p.geometry)),
    shapely.union_all(prot_p.geometry)), state))
plan_line = plan_p.geometry.iloc[0]
pln_parts = list(shapely.get_parts(plan_line))
defo_parts = list(shapely.get_parts(defo_all)) if defo_all.geom_type != 'Polygon' else [defo_all]
risk_parts = list(shapely.get_parts(risk)) if risk.geom_type in ('MultiPolygon','GeometryCollection') else [risk]
dg = gpd.GeoDataFrame(geometry=defo_parts, crs=P)
rp = gpd.GeoDataFrame(geometry=risk_parts, crs=P)

def draw_map(ax, xmin, ymin, xmax, ymax, show_state=True):
    if show_state:
        ax.add_patch(plt.Polygon(state.exterior.coords, closed=True,
                     facecolor='#F7F3E8', edgecolor='#333333', linewidth=2.2, zorder=1))
    dg.plot(ax=ax, color='#F2B58C', edgecolor='none', alpha=0.85, zorder=2, legend=False)
    prot_p.plot(ax=ax, color='#B5D99A', edgecolor='#7FAE5D', linewidth=0.4, alpha=0.9, zorder=3, legend=False)
    roads_p.plot(ax=ax, color='#555555', linewidth=0.35, zorder=4, legend=False)
    bx, by = buf.exterior.xy
    ax.plot(bx, by, color='#2C6FBB', linewidth=1.6, linestyle=':', zorder=5)
    rp.plot(ax=ax, color='#D7301F', edgecolor='#8B0000', linewidth=0.3, alpha=0.85, zorder=6, legend=False)
    for g in pln_parts:
        xx, yy = g.xy
        ax.plot(xx, yy, color='#FFD400', linewidth=3.2, zorder=7,
                path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()])
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal'); ax.axis('off')

def legend(ax):
    h = [
        mpatches.Patch(facecolor='#F2B58C', edgecolor='none', label='Deforested area'),
        mpatches.Patch(facecolor='#B5D99A', edgecolor='#7FAE5D', label='Protected forest'),
        mlines.Line2D([], [], color='#555555', linewidth=1.2, label='Existing roads'),
        mlines.Line2D([], [], color='#2C6FBB', linewidth=1.8, linestyle=':', label='5.5 km buffer (planned road)'),
        mpatches.Patch(facecolor='#D7301F', edgecolor='#8B0000', label='At risk of new deforestation (566.5 km²)'),
        mlines.Line2D([], [], color='#FFD400', linewidth=3.2,
                      path_effects=[pe.Stroke(linewidth=4.5, foreground='black'), pe.Normal()],
                      label='Planned road (Proposed BR-421, 208 km)'),
        mpatches.Patch(facecolor='#F7F3E8', edgecolor='#333333', label='Rondônia state boundary'),
    ]
    ax.legend(handles=h, loc='lower left', fontsize=8.5, framealpha=0.92)

fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.05)
ax1 = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])

sx0, sy0, sx1, sy1 = state.bounds
draw_map(ax1, sx0-15000, sy0-15000, sx1+15000, sy1+15000)
legend(ax1)
ax1.set_title('Rondônia – deforested area, protected forest & existing roads', fontsize=13, pad=8)

bx0, by0, bx1, by1 = buf.bounds
m = 30000
draw_map(ax2, bx0-m, by0-m, bx1+m, by1+m, show_state=False)
legend(ax2)
ax2.set_title('Zoom: planned BR-421 corridor & at-risk forest', fontsize=13, pad=8)

fig.suptitle('Effect of the planned BR-421 road on deforestation in Rondônia, Brazil\n'
             '5.5 km buffer: 2,383.5 km² total  |  at risk of new deforestation: 566.5 km² (23.8% of buffer)',
             fontsize=15, fontweight='bold', y=0.985)

def scale_bar(ax, x, y, length_m, label):
    ax.plot([x, x+length_m], [y, y], color='k', linewidth=3)
    for xx in (x, x+length_m/2, x+length_m):
        ax.plot([xx, xx], [y-2000, y+2000], color='k', linewidth=1.5)
    ax.text(x+length_m/2, y+8000, label, ha='center', fontsize=9)
scale_bar(ax1, sx0+40000, sy0+50000, 100000, '100 km')
scale_bar(ax2, bx0+12000, by0+14000, 20000, '20 km')

os.makedirs('pred_results', exist_ok=True)
fig.savefig('pred_results/predictedRiskyArea.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close(fig)

from PIL import Image
im = Image.open('pred_results/predictedRiskyArea.png')
print("saved:", im.size, os.path.getsize('pred_results/predictedRiskyArea.png'), "bytes")

