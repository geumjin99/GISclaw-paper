# Correct: the three factors are drawn as quantile classes via plot(color=[...]) (no column=), so only
# face colours identify them. A session of six steps run in order; step 4 raises an IndexError mid-way
# (its imports, figure and layers persist) and is wrapped in try/except.
import matplotlib
matplotlib.use('Agg')
# --- block 1 ---
import geopandas as gpd
bus_raw = gpd.read_file("dataset/BusServiceArea.geojson")
demo_raw = gpd.read_file("dataset/HamiltonDemographics.geojson")
print(bus_raw.crs, demo_raw.crs)
print("invalid:", (~bus_raw.is_valid).sum(), (~demo_raw.is_valid).sum())
bus_sa = bus_raw.to_crs(32616).copy()
demo = demo_raw.to_crs(32616).copy()
bus_sa["geometry"] = bus_sa.geometry.make_valid()
demo["geometry"] = demo.geometry.make_valid()
print("invalid after:", (~bus_sa.is_valid).sum(), (~demo.is_valid).sum())
print(len(bus_sa), len(demo), bus_sa.crs)
print(bus_sa.total_bounds, demo.total_bounds)
print(bus_sa.geom_type.value_counts(), demo.geom_type.value_counts())
print("bus area km2:", bus_sa.area.sum()/1e6)
# --- block 2 ---
import pandas as pd, numpy as np
for c in ['populationtotals_TOTPOP_CY','households_ACSHHBPOV','AtRisk_ACSOVEH0','AREALAND']:
    print(c, demo[c].dtype, demo[c].isna().sum())
pop = pd.to_numeric(demo['populationtotals_TOTPOP_CY'], errors='coerce')
land = pd.to_numeric(demo['AREALAND'], errors='coerce')
proj_km2 = demo.geometry.area/1e6
land_km2 = np.where((land>0)&land.notna(), land/1e6, proj_km2)
print("fallback used:", int(((land<=0)|land.isna()).sum()))
demo['land_km2'] = land_km2
demo['pop_density'] = pop/demo['land_km2']
demo['poverty'] = pd.to_numeric(demo['households_ACSHHBPOV'], errors='coerce')
demo['no_vehicle'] = pd.to_numeric(demo['AtRisk_ACSOVEH0'], errors='coerce')
cols=['pop_density','poverty','no_vehicle']
for c in cols: demo[c+'_missing']=demo[c].isna()
print(demo[cols].describe())
print(demo[cols].isna().sum())
# --- block 3 ---
import geopandas as gpd, numpy as np
bus_keep = bus_sa[['geometry']].copy()
demo['orig_km2'] = demo.geometry.area/1e6
served_blocks = gpd.overlay(demo, bus_keep, how='intersection', keep_geom_type=True)
served_blocks['served_km2'] = served_blocks.geometry.area/1e6
served_blocks['served_frac'] = served_blocks['served_km2']/served_blocks['orig_km2']
served_blocks['served_pop'] = served_blocks['populationtotals_TOTPOP_CY'].astype(float)*served_blocks['served_frac']
print(len(served_blocks), served_blocks.crs)
print('total served km2:', served_blocks.served_km2.sum(), ' bus_sa km2:', bus_sa.area.sum()/1e6)
print('block groups touched:', served_blocks.GEOID.nunique())
w = served_blocks.served_pop
for c in ['poverty','no_vehicle','pop_density']:
    print(c, 'pop-weighted mean inside:', np.average(served_blocks[c], weights=w))
# outside
out = gpd.overlay(demo, bus_keep, how='difference', keep_geom_type=True)
out['frac'] = (out.geometry.area/1e6)/out['orig_km2']
wo = out['populationtotals_TOTPOP_CY'].astype(float)*out['frac']
for c in ['poverty','no_vehicle','pop_density']:
    print(c, 'pop-weighted mean outside:', np.average(out[c], weights=wo))
print('served pop est:', w.sum(), ' outside pop est:', wo.sum())
print(served_blocks[['poverty','no_vehicle','pop_density','served_km2','served_frac']].describe())
# --- block 4 (raises an error) ---
try:
    import matplotlib.pyplot as plt, mapclassify, numpy as np
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    print(served_blocks[['pop_density','poverty','no_vehicle']].describe().loc[['count','min','max']])

    fig, ax = plt.subplots(figsize=(12,12))
    demo.plot(ax=ax, color='#e6e6e6', edgecolor='#999999', linewidth=0.3)
    layers = [('pop_density','Blues','Population density (people/km²)'),
              ('poverty','Reds','Households below poverty (ACSHHBPOV)'),
              ('no_vehicle','Greens','Households with no vehicle (ACSOVEH0)')]
    legends=[]; locs=['upper left','upper right','lower right']
    for (col,cmap,label),loc in zip(layers,locs):
        q = mapclassify.Quantiles(served_blocks[col], k=5)
        cm = plt.get_cmap(cmap)
        colors = [cm(0.25+0.75*i/4) for i in range(5)]
        served_blocks.assign(_cls=q.yb).plot(ax=ax, color=[colors[i] for i in q.yb], alpha=0.4, edgecolor='none')
        lo = [served_blocks[col].min()]+list(q.bins[:-1])
        handles=[Patch(facecolor=colors[i], alpha=0.6, edgecolor='k', lw=0.3,
                       label=f"{lo[i]:,.0f} – {q.bins[i]:,.0f}") for i in range(5)]
        leg = ax.legend(handles=handles, title=label+'\n(quantiles, served areas)', loc=loc, fontsize=8, title_fontsize=9, framealpha=0.85)
        ax.add_artist(leg); legends.append(leg)
        print('plotted', col, cmap, 'bins', np.round(q.bins,1))
    bus_sa.boundary.plot(ax=ax, color='black', linewidth=2)
    ax.legend(handles=[Line2D([0],[0],color='black',lw=2,label='10-min bus stop service area'),
                       Patch(facecolor='#e6e6e6',edgecolor='#999999',label='Hamilton County block groups')],
              loc='lower left', fontsize=9, framealpha=0.85)
    ax.annotate('N', xy=(0.95,0.12), xytext=(0.95,0.05), xycoords='axes fraction', ha='center', fontsize=14,
                fontweight='bold', arrowprops=dict(facecolor='black', width=4, headwidth=12))
    ax.set_title('Public Transit Access in Hamilton County, TN:\nBus Service Area overlaid with Population Density, Poverty and No-Vehicle Households', fontsize=14)
    ax.set_xlabel('Easting (m, UTM 16N / EPSG:32616)'); ax.set_ylabel('Northing (m, UTM 16N / EPSG:32616)')
    print('layers: base block groups, 3 choropleths, service-area outline; legends:', len(legends)+1)
except Exception as e:
    print('Error:', e)
# --- block 5 ---
plt.close(fig)
fig, ax = plt.subplots(figsize=(12,12))
demo.plot(ax=ax, color='#e6e6e6', edgecolor='#999999', linewidth=0.3)
legends=[]
for (col,cmap,label),loc in zip(layers,['upper left','upper right','lower right']):
    q = mapclassify.Quantiles(served_blocks[col], k=5)
    k = len(q.bins); cm = plt.get_cmap(cmap)
    colors = [cm(0.25+0.75*i/max(k-1,1)) for i in range(k)]
    served_blocks.plot(ax=ax, color=[colors[i] for i in q.yb], alpha=0.4, edgecolor='none')
    lo = [served_blocks[col].min()]+list(q.bins[:-1])
    handles=[Patch(facecolor=colors[i], alpha=0.6, edgecolor='k', lw=0.3,
                   label=f"{lo[i]:,.0f} – {q.bins[i]:,.0f}") for i in range(k)]
    leg = ax.legend(handles=handles, title=label+'\n(quantiles, served areas)', loc=loc, fontsize=8, title_fontsize=9, framealpha=0.85)
    ax.add_artist(leg); legends.append(leg)
    print('plotted', col, cmap, k, 'classes, bins', np.round(q.bins,1), 'class counts', np.bincount(q.yb))
bus_sa.boundary.plot(ax=ax, color='black', linewidth=2)
ax.legend(handles=[Line2D([0],[0],color='black',lw=2,label='10-min bus stop service area'),
                   Patch(facecolor='#e6e6e6',edgecolor='#999999',label='Hamilton County block groups')],
          loc='lower left', fontsize=9, framealpha=0.85)
ax.annotate('N', xy=(0.93,0.2), xytext=(0.93,0.13), xycoords='axes fraction', ha='center', fontsize=14,
            fontweight='bold', arrowprops=dict(facecolor='black', width=4, headwidth=12))
ax.set_title('Public Transit Access in Hamilton County, TN:\nBus Service Area overlaid with Population Density, Poverty and No-Vehicle Households', fontsize=14)
ax.set_xlabel('Easting (m, UTM 16N / EPSG:32616)'); ax.set_ylabel('Northing (m, UTM 16N / EPSG:32616)')
print('collections on axes:', len(ax.collections), '| legends:', len(legends)+1)
# --- block 6 ---
import os
os.makedirs('pred_results', exist_ok=True)
p='pred_results/transit_access.png'
fig.savefig(p, dpi=150, bbox_inches='tight')
print(p, os.path.exists(p), os.path.getsize(p))
import matplotlib.image as mpimg
im=mpimg.imread(p); print(im.shape, im.std())
