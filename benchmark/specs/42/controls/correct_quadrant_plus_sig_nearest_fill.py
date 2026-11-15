"""Correct: quadrant and significance are kept in two columns; the two group-less polygons are filled
from the nearest group. The script is a sequence of blocks run in order; blocks that raise an error
are wrapped in try/except.
"""

# --- block 1
hoods = gpd.read_file('dataset/berlin-neighbourhoods.geojson')
print(f"Rows: {len(hoods)}")
print(f"CRS: {hoods.crs.to_string() if hoods.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in hoods.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in hoods.total_bounds]}")
_cols = [c for c in hoods.columns if c != hoods.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(hoods[_cols].head(3).to_string())


# --- block 2 (wrapped in try/except)
try:
    exec('import pandas as pd\nlistings = pd.read_csv(\'dataset/berlin-listings.csv\', usecols=[\'id\',\'latitude\',\'longitude\',\'price\',\'price_currency\',\'number_of_reviews\'])\nprint(listings.shape)\nprint(listings[[\'latitude\',\'longitude\',\'price\']].head(10))\nprint()\nprint(listings[[\'latitude\',\'longitude\',\'price\']].describe())\nprint()\nprint("price missing:", listings[\'price\'].isna().sum())\nprint("price == 0:", (listings[\'price\']==0).sum())\nprint("lat missing:", listings[\'latitude\'].isna().sum(), "lng missing:", listings[\'longitude\'].isna().sum())\nprint("price_currency:", listings[\'price_currency\'].unique() if \'price_currency\' in listings else \'n/a\')\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 3
import pandas as pd
listings = pd.read_csv('dataset/berlin-listings.csv', nrows=5)
print(list(listings.columns))


# --- block 4
import pandas as pd
listings = pd.read_csv('dataset/berlin-listings.csv', usecols=['id','latitude','longitude','price','number_of_reviews'])
print(listings.shape)
print(listings[['latitude','longitude','price']].head(10))
print(listings['price'].dtype)
print("price sample uniques:", listings['price'].astype(str).str[:30].unique()[:15])
print("lat missing:", listings['latitude'].isna().sum(), "lng missing:", listings['longitude'].isna().sum())
print("lat range:", listings['latitude'].min(), listings['latitude'].max())
print("lng range:", listings['longitude'].min(), listings['longitude'].max())


# --- block 5
import geopandas as gpd
from shapely.geometry import Point

print("price missing:", listings['price'].isna().sum())
print("price <= 0:", (listings['price'] <= 0).sum())
print("price range:", listings['price'].min(), listings['price'].max())
print("price median:", listings['price'].median())

pts = gpd.GeoDataFrame(listings, geometry=gpd.points_from_xy(listings['longitude'], listings['latitude']), crs='EPSG:4326')
print(pts.shape, pts.crs)
# sanity: points inside Berlin bbox
in_bounds = pts.within(gpd.GeoSeries([gpd.GeoSeries([hoods.geometry.unary_union]).bounds]).to_crs(4326)) if False else None
print("points total:", len(pts))


# --- block 6 (wrapped in try/except)
try:
    exec('import geopandas as gpd\njoined = gpd.sjoin(pts[[\'id\',\'price\']], hoods[[\'neighbourhood\',\'neighbourhood_group\',\'geometry\']], how=\'left\', predicate=\'within\')\nprint("joined rows:", len(joined))\nprint("matched to a neighbourhood:", joined[\'neighbourhood\'].notna().sum())\nprint("unmatched (outside all polygons):", joined[\'neighbourhood\'].isna().sum())\nprint()\nprint("distinct neighbourhoods with listings:", joined[\'neighbourhood\'].nunique())\nprint("distinct groups with listings:", joined[\'neighbourhood_group\'].nunique())\nprint(joined.groupby(\'neighbourhood_group\')[\'price\'].agg([\'count\',\'median\']).head(15))\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 7
joined = gpd.sjoin(pts[['id','price','geometry']], hoods[['neighbourhood','neighbourhood_group','geometry']], how='left', predicate='within')
print("joined rows:", len(joined))
print("matched to a neighbourhood:", joined['neighbourhood'].notna().sum())
print("unmatched (outside all polygons):", joined['neighbourhood'].isna().sum())
print()
print("distinct neighbourhoods with listings:", joined['neighbourhood'].nunique())
print("distinct groups with listings:", joined['neighbourhood_group'].nunique())
print(joined.groupby('neighbourhood_group')['price'].agg(['count','median']).head(15))


# --- block 8
print("Neighbourhoods with missing fields:")
print(hoods[hoods['neighbourhood'].isna() | hoods['neighbourhood_group'].isna()][['neighbourhood','neighbourhood_group']])
print()
print("Group value counts (all 140):")
print(hoods['neighbourhood_group'].value_counts(dropna=False))
print()
print("Missing neighbourhood names:", hoods['neighbourhood'].isna().sum())
print("Missing neighbourhood_group:", hoods['neighbourhood_group'].isna().sum())


# --- block 9
import numpy as np

# Check the two rows with missing attributes
for i in [11, 12]:
    g = hoods.loc[i, 'geometry']
    print(f"row {i}: area={g.area:.6f}, bounds={g.bounds}, is_valid={g.is_valid}")


# --- block 10
from shapely import contains
groups_dissolved = hoods[hoods['neighbourhood_group'].notna()].dissolve(by='neighbourhood_group')['geometry']
print(groups_dissolved.index.tolist())

for i in [11, 12]:
    for grp, geom in groups_dissolved.items():
        if geom.contains(hoods.loc[i, 'geometry']) or geom.touches(hoods.loc[i, 'geometry']):
            print(f"row {i} is inside/touches group: {grp}")


# --- block 11
for i in [11, 12]:
    g = hoods.loc[i, 'geometry']
    for grp, geom in groups_dissolved.items():
        if geom.intersects(g):
            print(f"row {i} intersects group: {grp}")
    # also check which named neighbourhood polygons it overlaps
    for j, h in hoods.iterrows():
        if j not in (11,12) and h['geometry'].intersects(g):
            print(f"  row {i} also overlaps neighbourhood {h['neighbourhood']} (group {h['neighbourhood_group']})")


# --- block 12
for i in [11, 12]:
    g = hoods.loc[i, 'geometry']
    print(f"row {i}: centroid={g.centroid}, bounds={g.bounds}")
    overlaps = []
    for j, h in hoods.iterrows():
        if j != i and h['geometry'].intersects(g):
            overlaps.append((h['neighbourhood'], h['neighbourhood_group']))
    print("  overlaps with:", overlaps[:5])
    print()


# --- block 13
for i in [11, 12]:
    g = hoods.loc[i, 'geometry']
    print(f"row {i}:")
    for grp, geom in groups_dissolved.items():
        d = geom.distance(g)
        if d < 0.002:
            print(f"  dist to {grp}: {d:.6f}  intersects={geom.intersects(g)}  contains={geom.contains(g)}")


# --- block 14
# These sit in gaps (holes) between group polygons. Assign by closest group polygon (small gap).
for i in [11, 12]:
    g = hoods.loc[i, 'geometry']
    dists = {grp: geom.distance(g) for grp, geom in groups_dissolved.items()}
    best = min(dists, key=dists.get)
    print(f"row {i}: closest group = {best} (dist {dists[best]:.6f} deg); next: {sorted(dists.items(), key=lambda x: x[1])[1][0]} ({sorted(dists.items(), key=lambda x: x[1])[1][1]:.6f})")


# --- block 15
for i in [11, 12]:
    g = hoods.loc[i, 'geometry']
    named = hoods[hoods['neighbourhood'].notna()]
    d = named['geometry'].distance(g)
    near = d.nsmallest(3)
    for idx in near.index:
        print(f"row {i}: nearest named = {named.loc[idx,'neighbourhood']} (group={named.loc[idx,'neighbourhood_group']}, dist {near[idx]:.5f} deg)")
    print()


# --- block 16 (wrapped in try/except)
try:
    exec('import numpy as np\nimport geopandas as gpd\n\nanalysis = hoods[[\'neighbourhood\',\'neighbourhood_group\',\'geometry\']].copy()\n\n# Handle missing group for the 2 unnamed polygons: assign by closest group polygon (documented above)\nfor i in [11, 12]:\n    g = analysis.loc[i, \'geometry\']\n    best = min(groups_dissolved.index, key=lambda grp: groups_dissolved[grp].distance(g))\n    analysis.loc[i, \'neighbourhood_group\'] = best\n    print(f"assigned row {i} -> {best}")\n\nprint("\\nremaining missing groups:", analysis[\'neighbourhood_group\'].isna().sum())\n\n# Median price per group, assigned to every neighbourhood in that group\ngroup_median = joined.dropna(subset=[\'neighbourhood_group\']).groupby(\'neighbourhood_group\')[\'price\'].median()\nprint("\\nGroup medians:")\nprint(group_median)\n\nanalysis[\'median_price\'] = analysis[\'neighbourhood_group\'].map(group_median)\nprint("\\nmedian_price missing:", analysis[\'median_price\'].isna().sum())\nprint(analysis[\'median_price\'].describe())\nanalysis.to_crs? print(analysis.crs)\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 17
import numpy as np
import geopandas as gpd

analysis = hoods[['neighbourhood','neighbourhood_group','geometry']].copy()

for i in [11, 12]:
    g = analysis.loc[i, 'geometry']
    best = min(groups_dissolved.index, key=lambda grp: groups_dissolved[grp].distance(g))
    analysis.loc[i, 'neighbourhood_group'] = best
    print(f"assigned row {i} -> {best}")

print("\nremaining missing groups:", analysis['neighbourhood_group'].isna().sum())

group_median = joined.dropna(subset=['neighbourhood_group']).groupby('neighbourhood_group')['price'].median()
print("\nGroup medians:")
print(group_median)

analysis['median_price'] = analysis['neighbourhood_group'].map(group_median)
print("\nmedian_price missing:", analysis['median_price'].isna().sum())
print(analysis['median_price'].describe())
print("\nCRS:", analysis.crs)
print("rows:", len(analysis))


# --- block 18 (wrapped in try/except)
try:
    exec('import libpysal as lps\n\nw = lps.weights.Queen(analysis[\'geometry\'])\nprint("n:", w.n)\nprint("n_neighbors:", w.n_neighbors)\nprint("is_symmetric:", w.is_symmetric)\nprint("has_islands:", w.islands)\nprint("num islands:", w.num_islands)\nprint("sum of weights:", w.sum())\nprint("min/max neighbours:", w.nn.min(), w.nn.max())\nw.plot_transform = \'R\'\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 19 (wrapped in try/except)
try:
    exec('print("n:", w.n)\nprint("is_symmetric:", w.is_symmetric)\nprint("num_islands:", w.num_islands)\nprint("min/max neighbours:", w.min_neighbors, w.max_neighbors)\nprint("mean neighbours:", w.mean_neighbors)\nprint("sum of weights:", w.sum())\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 20 (wrapped in try/except)
try:
    exec('print("n:", w.n)\nprint("asymmetry:", w.asymmetry())\nprint("num_islands:", w.num_islands)\nprint("min/max neighbours:", w.min_neighbors, w.max_neighbors)\nprint("mean neighbours:", w.mean_neighbors)\nprint("sum of weights:", w.sum())\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 21 (wrapped in try/except)
try:
    exec('print("n:", w.n)\nprint("islands:", w.islands())\nprint("asymmetry pairs:", w.asymmetry())\nprint("min/max neighbours:", w.min_neighbors, w.max_neighbors)\nprint("mean neighbours:", w.mean_neighbors)\nprint("sum of weights:", w.sum())\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 22 (wrapped in try/except)
try:
    exec('print("islands:", w.islands)\nprint("asymmetry:", w.asymmetry)\nprint("min/max neighbours:", w.min_neighbors, w.max_neighbors)\nprint("mean neighbours:", w.mean_neighbors)\nprint("sum of weights:", w.sum)\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 23
from esda.moran import Moran, Moran_Local
import numpy as np

y = analysis['median_price'].values
print("y shape:", y.shape, "unique:", np.unique(y))

m_global = Moran(y, w, transformation='r', permutations=999, two_tailed=True)
print("\nGlobal Moran's I:")
print("  I =", round(m_global.I, 4))
print("  E[I] =", m_global.EI)
print("  z-score =", round(m_global.z_sim, 3))
print("  p-value (sim) =", round(m_global.p_sim, 4))

m_local = Moran_Local(y, w, transformation='r', permutations=999, geoda_quads=True, n_jobs=4, seed=42)
print("\nLocal Moran computed. sim type:", m_local.__class__.__name__)
print("  attributes:", [a for a in m_local.__dict__ if not a.startswith('_')])


# --- block 24 (wrapped in try/except)
try:
    exec('import pandas as pd\n\ndf = pd.DataFrame({\n    \'neighbourhood\': analysis[\'neighbourhood\'].values,\n    \'neighbourhood_group\': analysis[\'neighbourhood_group\'].values,\n    \'median_price\': analysis[\'median_price\'].values,\n    \'z\': m_local.z,\n    \'Is\': m_local.Is,\n    \'quadrant\': m_local.quads,\n    \'p_sim\': m_local.p_sim,\n    \'p_z_sim\': m_local.p_z_sim,\n})\nprint("Quadrant counts:")\nprint(df[\'quadrant\'].value_counts(dropna=False))\nprint()\n# significant at alpha=0.10\nsig = df[\'p_sim\'] < 0.10\nprint("significant (p_sim<0.10):", sig.sum())\nprint(df[sig][\'quadrant\'].value_counts())\nprint()\nprint("  significant p<0.05:", (df[\'p_sim\']<0.05).sum())\nprint(df[(df[\'p_sim\']<0.05)][\'quadrant\'].value_counts())\nprint()\nprint(df.describe())\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 25
print("z:", len(m_local.z), "Is:", len(m_local.Is), "quads:", len(m_local.quads) if m_local.quads is not None else None, "p_sim:", len(m_local.p_sim))
print("type quads:", type(m_local.quads))
print("quads sample:", m_local.quads[:20] if m_local.quads is not None else None)
print("p_sim sample:", m_local.p_sim[:20])


# --- block 26
import pandas as pd
print("q:", m_local.q[:20], "len:", len(m_local.q))
print("q unique:", np.unique(m_local.q))
# quadrant label mapping: 1=HH, 2=HL, 3=LL, 4=LH
qmap = {1:'HH (hot)', 2:'HL (outlier)', 3:'LL (cold)', 4:'LH (outlier)'}

df = pd.DataFrame({
    'neighbourhood': analysis['neighbourhood'].values,
    'neighbourhood_group': analysis['neighbourhood_group'].values,
    'median_price': analysis['median_price'].values,
    'z': m_local.z,
    'Is': m_local.Is,
    'quadrant': [qmap[v] for v in m_local.q],
    'p_sim': m_local.p_sim,
})
print("\nQuadrant counts:")
print(df['quadrant'].value_counts())
print("\nSignificant p_sim<0.10:", (df['p_sim']<0.10).sum())
print(df[df['p_sim']<0.10]['quadrant'].value_counts())
print("\nSignificant p_sim<0.05:", (df['p_sim']<0.05).sum())
print(df[df['p_sim']<0.05]['quadrant'].value_counts())
print("\ndf head:")
print(df.head(8).to_string())


# --- block 27
import numpy as np
z = m_local.z
Is = m_local.Is
q = m_local.q

# Sign-based standard quadrant classification
def quad_from_signs(z_i, Is_i):
    # spatial lag of z = Is / z
    if z_i > 0 and Is_i > 0: return 'HH'
    if z_i < 0 and Is_i > 0: return 'LL'
    if z_i > 0 and Is_i < 0: return 'HL'
    if z_i < 0 and Is_i < 0: return 'LH'

sign_q = np.array([quad_from_signs(z[i], Is[i]) for i in range(len(z))])
code_to_label = {}
for i in range(len(z)):
    code_to_label.setdefault(q[i], sign_q[i])
print("Mapping q-code -> sign-based quadrant:")
for k in sorted(code_to_label):
    # ensure consistency
    vals = set(sign_q[q==k])
    print(f"  q={k} -> {vals}")
print()
print("Counts from sign-based:")
print(pd.Series(sign_q).value_counts())


# --- block 28
import pandas as pd, numpy as np

# Correct quadrant mapping verified against sign(z) and sign(Is)
qmap = {1:'HH', 2:'LL', 3:'LH', 4:'HL'}
df = pd.DataFrame({
    'neighbourhood': analysis['neighbourhood'].values,
    'neighbourhood_group': analysis['neighbourhood_group'].values,
    'median_price': analysis['median_price'].values,
    'z': m_local.z,
    'Is': m_local.Is,
    'quadrant': [qmap[v] for v in m_local.q],
    'p_sim': m_local.p_sim,
})
df['significant'] = df['p_sim'] < 0.05
print("Total neighbourhoods:", len(df))
print("\nQuadrant counts (all):")
print(df['quadrant'].value_counts())
print("\nSignificant (p_sim<0.05) by quadrant:")
print(df[df['significant']]['quadrant'].value_counts())
print("\nSignificant (p_sim<0.10) by quadrant:")
print(df[df['p_sim']<0.10]['quadrant'].value_counts())
print("\nHot spots (HH & sig p<0.05):", ((df['quadrant']=='HH')&df['significant']).sum())
print("Cold spots (LL & sig p<0.05):", ((df['quadrant']=='LL')&df['significant']).sum())
print("\nGlobal Moran's I: I=%.4f  z_sim=%.3f  p_sim=%.4f" % (m_global.I, m_global.z_sim, m_global.p_sim))
print("mean price = %.3f, sd = %.3f" % (df['median_price'].mean(), df['median_price'].std()))


# --- block 29
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

proj = analysis.to_crs('EPSG:32633')
proj_df = df.join(proj.set_index(analysis.index)[['geometry']])

# group boundaries
group_diss = hoods[hoods['neighbourhood_group'].notna()].dissolve(by='neighbourhood_group')['geometry'].to_crs('EPSG:32633')

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Spatial Autocorrelation of Airbnb Median Prices in Berlin (Local Moran\'s I)', fontsize=15, fontweight='bold')

# ---- Panel 1: median price choropleth ----
ax = axes[0, 0]
p1 = proj.plot(column='median_price', ax=ax, cmap='YlOrRd', legend=True, edgecolor='white', linewidth=0.4,
               legend_kwds={'label': 'Median price (€/night)', 'fmt': '%.0f'})
for g, geom in group_diss.items():
    ax.plot(*geom.exterior.xy if geom.geom_type=='Polygon' else zip(*[list(c) for c in geom.bounds]) , color='k')  # placeholder
axes[0,0].set_title('a) Median Airbnb price per neighbourhood group', fontsize=12)
axes[0,0].set_axis_off()
axes[0,0].clear()
p1 = proj.plot(column='median_price', ax=axes[0,0], cmap='YlOrRd', edgecolor='white', linewidth=0.4,
               legend=True, legend_kwds={'label': 'Median price (€/night)', 'fmt': '%.0f'})
for geom in group_diss:
    ax2 = axes[0,0]
    x, y = geom.exterior.xy if geom.geom_type == 'Polygon' else geom.interiors[0].xy if geom.interiors else ([],[])
    if geom.geom_type == 'MultiPolygon':
        for p in geom.geoms:
            x, y = p.exterior.xy
            ax2.plot(x, y, color='k', ls='--', lw=1.2, alpha=0.7)
    else:
        ax2.plot(x, y, color='k', ls='--', lw=1.2, alpha=0.7)
axes[0,0].set_title("a) Median Airbnb price per neighbourhood group\n(group boundaries: dashed)", fontsize=11)
axes[0,0].set_axis_off()


# --- block 30
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import geopandas as gpd

plt.close('all')
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("Spatial Autocorrelation of Airbnb Prices in Berlin\n"
             "Global Moran's I = 0.697 (z = 13.37, p < 0.001)  |  Queen contiguity  |  group median prices",
             fontsize=14, fontweight='bold')

# group boundaries for reference
group_diss = hoods.dissolve(by='neighbourhood_group')['geometry'].to_crs('EPSG:32633')

def draw_group_bounds(ax):
    for geom in group_diss:
        polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
        for p in polys:
            x, y = p.exterior.xy
            ax.plot(x, y, color='k', ls='--', lw=1.0, alpha=0.6, zorder=5)

# ---- Panel a: median price choropleth ----
ax = axes[0, 0]
proj.plot(column='median_price', ax=ax, cmap='YlOrRd', edgecolor='white', linewidth=0.4,
          legend=True, legend_kwds={'label': 'Median price (€/night)', 'shrink': 0.8, 'fmt': '%.0f'})
draw_group_bounds(ax)
ax.set_title("a) Median Airbnb price (assigned by neighbourhood group)", fontsize=11)
ax.set_axis_off()

# ---- Panel b: Local Moran's I with significant hot/cold spots ----
ax = axes[0, 1]
color_map = {
    'HH': '#d73027',   # significant hot = dark red
    'LL': '#4575b4',   # significant cold = dark blue
}
# base: non-significant in light grey
sig = df['significant']
proj_sig = proj[sig.values]
proj_ns  = proj[~sig.values]
proj_ns.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.3, zorder=1)
# significant spots
for qval in ['HH', 'LL']:
    m = (sig & (df['quadrant'] == qval)).values
    proj[m].plot(ax=ax, color=color_map[qval], edgecolor='white', linewidth=0.3, zorder=3)
draw_group_bounds(ax)
handles = [
    Line2D([], [], color='#d73027', lw=6, label='Significant hot spot (HH, p<0.05)'),
    Line2D([], [], color='#4575b4', lw=6, label='Significant cold spot (LL, p<0.05)'),
    Line2D([], [], color='lightgrey', lw=6, label='Not significant (p≥0.05)'),
]
ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.9)
ax.set_title("b) Local Moran's I: significant clusters of median price", fontsize=11)
ax.set_axis_off()

plt.savefig('pred_results/moran_local.png', dpi=150, bbox_inches='tight')
plt.close()
print("saved part 1")


# --- block 31 (wrapped in try/except)
try:
    exec('import matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nfrom matplotlib.lines import Line2D\nimport numpy as np\n\nplt.close(\'all\')\nfig, axes = plt.subplots(2, 2, figsize=(16, 14))\nfig.suptitle("Spatial Autocorrelation of Airbnb Prices in Berlin\\n"\n             "Global Moran\'s I = 0.697 (z = 13.37, p < 0.001)  |  Queen contiguity  |  group median prices",\n             fontsize=14, fontweight=\'bold\')\n\ngroup_diss = hoods.dissolve(by=\'neighbourhood_group\')[\'geometry\'].to_crs(\'EPSG:32633\')\ndef draw_group_bounds(ax):\n    for geom in group_diss:\n        polys = list(geom.geoms) if geom.geom_type == \'MultiPolygon\' else [geom]\n        for p in polys:\n            x, y = p.exterior.xy\n            ax.plot(x, y, color=\'k\', ls=\'--\', lw=1.0, alpha=0.6, zorder=5)\n\n# ---- Panel a: median price choropleth ----\nax = axes[0, 0]\nproj.plot(column=\'median_price\', ax=ax, cmap=\'YlOrRd\', edgecolor=\'white\', linewidth=0.4,\n          legend=True, legend_kwds={\'label\': \'Median price (€/night)\', \'shrink\': 0.8, \'fmt\': \'%.0f\'})\ndraw_group_bounds(ax)\nax.set_title("a) Median Airbnb price (assigned by neighbourhood group)", fontsize=11)\nax.set_axis_off()\n\n# ---- Panel b: significant Local Moran clusters ----\nax = axes[0, 1]\nsig = df[\'significant\']\ncolor_map = {\'HH\': \'#d73027\', \'LL\': \'#4575b4\'}\nproj[~sig.values].plot(ax=ax, color=\'#e8e8e8\', edgecolor=\'white\', linewidth=0.3, zorder=1)\nfor qval in [\'HH\', \'LL\']:\n    m = (sig & (df[\'quadrant\'] == qval)).values\n    proj[m].plot(ax=ax, color=color_map[qval], edgecolor=\'white\', linewidth=0.3, zorder=3)\ndraw_group_bounds(ax)\nhandles = [\n    Line2D([], [], color=\'#d73027\', lw=6, label=\'Significant hot spot (HH, p<0.05) — n=37\'),\n    Line2D([], [], color=\'#4575b4\', lw=6, label=\'Significant cold spot (LL, p<0.05) — n=23\'),\n    Line2D([], [], color=\'#e8e8e8\', lw=6, label=\'Not significant (p≥0.05) — n=80\'),\n]\nax.legend(handles=handles, loc=\'lower left\', fontsize=8.5, framealpha=0.95)\nax.set_title("b) Local Moran\'s I: significant clusters of median price", fontsize=11)\nax.set_axis_off()\n\n# ---- Panel c: Moran scatterplot ----\nax = axes[1, 0]\nz = m_local.z\nsc = ax.scatter(z, m_local.Is, c=df[\'median_price\'], cmap=\'YlOrRd\', s=28, alpha=0.85, edgecolor=\'k\', linewidth=0.3)\ncb = fig.colorbar(sc, ax=ax, shrink=0.85)\ncb.set_label(\'Median price (€/night)\', fontsize=10)\n# 45-degree reference line through origin\nzlim = max(abs(z).min(), 0.5)\nax.plot([-1.2, 1.3], [-1.2, 1.3], \'k--\', lw=1, alpha=0.7)\nax.axhline(0, color=\'grey\', lw=0.5); ax.axvline(0, color=\'grey\', lw=0.5)\nax.set_xlabel(\'Standardised price (z-score of median price)\', fontsize=10)\nax.set_ylabel(\'Neighbour mean, standardised (L(z)z)\', fontsize=10)\nax.set_title("c) Moran scatterplot (quadrants)", fontsize=11)\n# quadrant text\nax.text(0.85, 0.98, \'HH (high price,\\nhigh neighbour)\', transform=ax.transAxes, ha=\'right\', va=\'top\', fontsize=8, color=\'#d73027\')\nax.text(0.15, 0.98, \'LH\', transform=ax.transAxes, ha=\'left\', va=\'top\', fontsize=8, color=\'grey\')\nax.text(0.15, 0.02, \'LL (low price,\\nlow neighbour)\', transform=ax.transAxes, ha=\'left\', va=\'bottom\', fontsize=8, color=\'#4575b4\')\nax.text(0.85, 0.02, \'HL\', transform=ax.transAxes, ha=\'right\', va=\'bottom\', fontsize=8, color=\'grey\')\n\n# ---- Panel d: Local Moran\'s I (Is) signed map ----\nax = axes[1, 1]\nproj.plot(column=\'Is\', ax=ax, cmap=\'RdBu\', edgecolor=\'white\', linewidth=0.4,\n          legend=True, legend_kwds={\'label\': "Local Moran\'s I (Is)", \'shrink\': 0.8})\ndraw_group_bounds(ax)\nax.set_title("d) Local Moran\'s I statistic (Is) per neighbourhood", fontsize=11)\nax.set_axis_off()\n\nplt.tight_layout(rect=[0, 0, 1, 0.95])\nplt.savefig(\'pred_results/moran_local.png\', dpi=150, bbox_inches=\'tight\')\nplt.close()\nprint("figure saved")\n', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 32
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.close('all')
# Build a projected GeoDataFrame carrying all analytic columns
proj = analysis.to_crs('EPSG:32633').copy()
proj['median_price'] = df['median_price'].values
proj['quadrant']     = df['quadrant'].values
proj['p_sim']        = df['p_sim'].values
proj['Is']           = m_local.Is
proj['significant']  = df['significant'].values
print("proj columns:", list(proj.columns))
print("Is range:", proj['Is'].min(), proj['Is'].max())


# --- block 33
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.close('all')
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle("Spatial Autocorrelation of Airbnb Prices in Berlin\n"
             "Global Moran's I = 0.697 (z = 13.37, p < 0.001)  |  Queen contiguity  |  group median prices",
             fontsize=14, fontweight='bold')

group_diss = hoods.dissolve(by='neighbourhood_group')['geometry'].to_crs('EPSG:32633')
def draw_group_bounds(ax):
    for geom in group_diss:
        polys = list(geom.geoms) if geom.geom_type == 'MultiPolygon' else [geom]
        for p in polys:
            x, y = p.exterior.xy
            ax.plot(x, y, color='k', ls='--', lw=1.0, alpha=0.6, zorder=5)

# ---- Panel a: median price choropleth ----
ax = axes[0, 0]
proj.plot(column='median_price', ax=ax, cmap='YlOrRd', edgecolor='white', linewidth=0.4,
          legend=True, legend_kwds={'label': 'Median price (€/night)', 'shrink': 0.8, 'fmt': '%.0f'})
draw_group_bounds(ax)
ax.set_title("a) Median Airbnb price (assigned by neighbourhood group)", fontsize=11)
ax.set_axis_off()

# ---- Panel b: significant Local Moran clusters ----
ax = axes[0, 1]
sig = proj['significant'].values
color_map = {'HH': '#d73027', 'LL': '#4575b4'}
proj[~sig].plot(ax=ax, color='#e8e8e8', edgecolor='white', linewidth=0.3, zorder=1)
for qval in ['HH', 'LL']:
    m = (sig & (proj['quadrant'] == qval).values)
    proj[m].plot(ax=ax, color=color_map[qval], edgecolor='white', linewidth=0.3, zorder=3)
draw_group_bounds(ax)
handles = [
    Line2D([], [], color='#d73027', lw=6, label='Significant hot spot (HH, p<0.05) — n=37'),
    Line2D([], [], color='#4575b4', lw=6, label='Significant cold spot (LL, p<0.05) — n=23'),
    Line2D([], [], color='#e8e8e8', lw=6, label='Not significant (p≥0.05) — n=80'),
]
ax.legend(handles=handles, loc='lower left', fontsize=8.5, framealpha=0.95)
ax.set_title("b) Local Moran's I: significant clusters of median price", fontsize=11)
ax.set_axis_off()

# ---- Panel c: Moran scatterplot ----
ax = axes[1, 0]
z = m_local.z
sc = ax.scatter(z, m_local.Is, c=df['median_price'], cmap='YlOrRd', s=28, alpha=0.85, edgecolor='k', linewidth=0.3)
cb = fig.colorbar(sc, ax=ax, shrink=0.85)
cb.set_label('Median price (€/night)', fontsize=10)
ax.plot([-1.2, 1.3], [-1.2, 1.3], 'k--', lw=1, alpha=0.7)
ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
ax.set_xlabel('Standardised price (z-score of median price)', fontsize=10)
ax.set_ylabel('Neighbour mean, standardised (L(z)z)', fontsize=10)
ax.set_title("c) Moran scatterplot (quadrants)", fontsize=11)
ax.text(0.97, 0.98, 'HH (high price,\nhigh neighbour)', transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#d73027')
ax.text(0.03, 0.98, 'LH', transform=ax.transAxes, ha='left', va='top', fontsize=8, color='grey')
ax.text(0.03, 0.02, 'LL (low price,\nlow neighbour)', transform=ax.transAxes, ha='left', va='bottom', fontsize=8, color='#4575b4')
ax.text(0.97, 0.02, 'HL', transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='grey')

# ---- Panel d: Local Moran's I (Is) signed map ----
ax = axes[1, 1]
proj.plot(column='Is', ax=ax, cmap='RdBu', edgecolor='white', linewidth=0.4,
          legend=True, legend_kwds={'label': "Local Moran's I (Is)", 'shrink': 0.8})
draw_group_bounds(ax)
ax.set_title("d) Local Moran's I statistic (Is) per neighbourhood", fontsize=11)
ax.set_axis_off()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('pred_results/moran_local.png', dpi=150, bbox_inches='tight')
plt.close()
print("figure saved")


# --- block 34
from PIL import Image
import os
import numpy as np
p = 'pred_results/moran_local.png'
print("exists:", os.path.exists(p), "size KB:", os.path.getsize(p)//1024)
im = np.array(Image.open(p).convert('RGB'))
print("shape:", im.shape)
print("unique-ish colors:", len(np.unique(im.reshape(-1,3), axis=0)))
print("std of pixels:", im.std().round(1))
# fraction of pixels that are not near-white
nonwhite = (im.sum(axis=2) < 700).mean()
print("non-white pixel fraction:", round(nonwhite, 3))


# --- block 35
print("="*70)
print("FINAL ANALYSIS SUMMARY")
print("="*70)
print(f"Neighbourhoods loaded:            {len(hoods)}")
print(f"Listings loaded:                  {len(listings):,}")
print(f"Listings matched to neighbourhood: {joined['neighbourhood'].notna().sum():,}")
print(f"Distinct neighbourhoods w/ listings: {joined['neighbourhood'].nunique()}")
print(f"Neighbourhood groups:             {joined['neighbourhood_group'].nunique()}")
print(f"Queen weights: n={w.n}, islands={w.islands}, min/max nb={w.min_neighbors}/{w.max_neighbors}")
print(f"Global Moran's I = {m_global.I:.4f}  (E[I]={m_global.EI:.4f})")
print(f"  z_sim = {m_global.z_sim:.3f}, p_sim = {m_global.p_sim:.4f}")
print(f"Quadrants: " + str(dict(df['quadrant'].value_counts())))
print(f"Significant (p<0.05): total={int(df['significant'].sum())}  HH(hot)={( (df['quadrant']=='HH')&df['significant']).sum()}  LL(cold)={((df['quadrant']=='LL')&df['significant']).sum()}")
print(f"median_price range: {df['median_price'].min()} - {df['median_price'].max()} (matches listings 16-241? no, group medians 35-50)")
print(f"missing median_price: {int(df['median_price'].isna().sum())}")
print()
print("Deliverable: pred_results/moran_local.png  (2x2 figure: price choropleth,")
print("  significant clusters, Moran scatterplot, Local I map)")

