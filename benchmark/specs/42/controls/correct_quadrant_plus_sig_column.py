"""Correct: the LISA quadrant is kept in a 'category' column and the significance in a separate
boolean 'significant' column. The script is a sequence of blocks run in order; blocks that raise an
error are wrapped in try/except.
"""

# --- block 1
import pandas as pd
import geopandas as gpd

# Look at the neighbourhoods file
neigh = gpd.read_file('dataset/berlin-neighbourhoods.geojson')
print(neigh.head())
print(neigh.columns.tolist())
print(neigh.crs)
print(neigh.shape)
print(neigh['neighbourhood_group'].unique()[:20])

# --- block 2
import pandas as pd
listings = pd.read_csv('dataset/berlin-listings.csv')
print(listings.shape)
print(listings.columns.tolist())
print(listings.head())
print(listings['price'].describe())
print('NaN prices:', listings['price'].isna().sum())
print(listings[['latitude','longitude']].describe())

# --- block 3
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

# Check price dtype and neighbourhood group NAs
print(listings['price'].dtype)
print(neigh['neighbourhood_group'].isna().sum())
print(neigh[neigh['neighbourhood_group'].isna()])

# Create listing points
points = gpd.GeoDataFrame(
    listings[['id','price','latitude','longitude']],
    geometry=gpd.points_from_xy(listings['longitude'], listings['latitude']),
    crs='EPSG:4326'
)
print(points.shape)

# Spatial join
joined = gpd.sjoin(points, neigh[['neighbourhood','neighbourhood_group','geometry']], how='inner', predicate='within')
print(joined.shape)
print(joined.columns.tolist())

# --- block 4
# Calculate median price per neighbourhood group
group_median = joined.groupby('neighbourhood_group')['price'].median()
print(group_median)

# Assign group median to each neighbourhood
neigh = neigh.dropna(subset=['neighbourhood_group']).copy()
neigh['median_price'] = neigh['neighbourhood_group'].map(group_median)
print('\nNeighbourhoods with data:', neigh['median_price'].notna().sum(), 'of', len(neigh))
print('Rows with missing median after assignment:', neigh['median_price'].isna().sum())

# Handle missing data: drop neighbourhoods without a median price
neigh = neigh.dropna(subset=['median_price']).reset_index(drop=True)
print(neigh.shape)
print(neigh.head())

# Save intermediate results
import os
os.makedirs('pred_results', exist_ok=True)
neigh.to_file('pred_results/neigh_with_median.geojson', driver='GeoJSON')

# --- block 5 (wrapped in try/except)
try:
    exec("import libpysal\nfrom esda.moran import Moran_Local\n\n# Build Queen contiguity weights\nw = libpysal.weights.Queen(neigh)\nw = w.symmetrize()\nw.transform = 'R'  # row-standardize for Moran's I\nw.to_shapefile('pred_results/neigh_weights.shp')\n\n# Global Moran's I (for reference)\nz = neigh['median_price'].values\nmi = libpysal.esda.moran.Moran(z, w)\nprint('Global Moran I:', round(mi.I, 4), 'p-value:', round(mi.p_sim, 4), 'exp:', round(mi.EI, 4))\n\n# Local Moran's I\nmoran = libpysal.esda.moran.Moran_Local(z, w, ztype='F', transforms='r')\nprint(moran)", globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 6
# Check geometry types and validity
print(neigh.geometry.geom_type.value_counts())
print(neigh.geometry.is_valid.value_counts())
bad = neigh[~neigh.geometry.is_valid]
print(bad[['neighbourhood','neighbourhood_group']].to_string())

# Inspect any invalid geometry
for i, g in neigh.geometry.items():
    if not g.is_valid:
        print(i, type(g), g.geom_type)
        if g.geom_type == 'MultiPolygon':
            for p in g.geoms:
                print('  part type:', type(p), getattr(p, 'geom_type', None), getattr(p, 'is_valid', None))

# --- block 7
import libpysal, shapely, inspect
print('libpysal', libpysal.__version__, 'shapely', shapely.__version__)
import libpysal.weights._contW_lists as c
print(inspect.getsource(c._get_boundary_points))
print(inspect.getsource(c._get_verts))

# --- block 8
import inspect
import libpysal.weights._contW_lists as c
print(inspect.getsource(c.ContiguityWeightsLists.__init__))
print(inspect.getsource(c.ContiguityWeightsLists.jcontiguity)[:2000])

# --- block 9
from libpysal.weights._contW_lists import _get_verts, _get_boundary_points
# Test on the first geometry
g0 = neigh.geometry.iloc[0]
print(type(g0))
try:
    v = _get_verts(g0)
    print('verts ok', len(v))
except Exception as e:
    print('ERR', e)

# iterate to find which one fails
for i, g in enumerate(neigh.geometry):
    try:
        _get_verts(g)
    except Exception as e:
        print('fails at', i, type(g), e)
        print('geoms types:', [type(x) for x in g.geoms])
        break

# --- block 10
import inspect
import libpysal.weights.contiguity as ct
src = inspect.getsource(ct._build)
print(src[:1500])
# check where collection comes from
src2 = inspect.getsource(ct.Queen.__init__)
print(src2)

# --- block 11
items = list(neigh)
print(type(items[0]))
print(items[0][:200] if isinstance(items[0], str) else items[0])

# try passing the geometry column explicitly
w = libpysal.weights.Queen(neigh.geometry)
print('Queen weights built OK')

# --- block 12 (wrapped in try/except)
try:
    exec("import libpysal\nfrom esda.moran import Moran\n\nw = libpysal.weights.Queen(neigh.geometry, byid=False)\nw = w.symmetrize()\nw.transform = 'R'\n\nz = neigh['median_price'].values\nmi = Moran(z, w)\nprint('Global Moran I:', round(mi.I, 4), '| p_sim:', round(mi.p_sim, 4), '| E(I):', round(mi.EI, 4), '| SD(I):', round(mi.SDI, 4), '| z:', round(mi.z_sim, 2))\nprint('n:', w.n, 'neighbors:', len(w.neighbors))\n\nmoran_local = Moran_Local(z, w, ztype='F', transforms='r')\nprint('\\nLocal Moran object:')\nprint('I:', moran_local.I.shape, 'z_sim:', moran_local.z_sim.shape)\nneigh['moran_I'] = moran_local.I\nneigh['moran_z'] = moran_local.z_sim\nneigh['moran_p_sim'] = moran_local.p_sim\nprint(neigh[['neighbourhood','neighbourhood_group','median_price','moran_I','moran_z','moran_p_sim']].head(10))", globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 13 (wrapped in try/except)
try:
    exec("w = libpysal.weights.Queen(neigh.geometry)\nw = w.symmetrize()\nw.transform = 'R'\n\nz = neigh['median_price'].values\nmi = Moran(z, w)\nprint('Global Moran I:', round(mi.I, 4), '| p_sim:', round(mi.p_sim, 4), '| E(I):', round(mi.EI, 4), '| SD(I):', round(mi.SDI, 4), '| z:', round(mi.z_sim, 2))\nprint('n:', w.n, 'unique neighbors:', len(w.neighbors))\n\nmoran_local = Moran_Local(z, w, ztype='F', transforms='r')\nneigh['moran_I'] = moran_local.I\nneigh['moran_z'] = moran_local.z_sim\nneigh['moran_p_sim'] = moran_local.p_sim\nprint(neigh[['neighbourhood','neighbourhood_group','median_price','moran_I','moran_z','moran_p_sim']].head(10).to_string())", globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 14
print([a for a in dir(mi) if not a.startswith('_')])
print('I:', mi.I)
print('EI:', mi.EI)
print('SD:', mi.SD if hasattr(mi,'SD') else 'n/a')
print('z_sim:', mi.z_sim if hasattr(mi,'z_sim') else 'n/a')
print('p_sim:', mi.p_sim)

# --- block 15 (wrapped in try/except)
try:
    exec("moran_local = Moran_Local(z, w, ztype='F', transforms='r')\nneigh['moran_I'] = moran_local.I\nneigh['moran_z'] = moran_local.z_sim\nneigh['moran_p_sim'] = moran_local.p_sim\n\nneigh['moran_I'].head()\npd_df = neigh[['neighbourhood','neighbourhood_group','median_price','moran_I','moran_z','moran_p_sim']].head(10)\nprint(pd_df.to_string())\n\n# significance at 95% (two-sided, p < 0.05)\nsig = neigh['moran_p_sim'] < 0.05\nprint('\\nsignificant count:', sig.sum())\n\n# Quadrant classification: I sign vs z-value sign (high vs low)\nhigh_z = neigh['moran_z'] > 0\nneigh['category'] = np.where((neigh['moran_I'] > 0) & (high_z), 'HH',\n                     np.where((neigh['moran_I'] < 0) & (high_z), 'LH',\n                     np.where((neigh['moran_I'] > 0) & (~high_z), 'HL',\n                               'LL')))\nprint(neigh.groupby(['category'])['neighbourhood'].count())\nprint('\\nHH (hot spots):')\nprint(neigh[(neigh['category']=='HH') & sig][['neighbourhood','median_price','moran_I']].to_string())\nprint('\\nLL (cold spots):')\nprint(neigh[(neigh['category']=='LL') & sig][['neighbourhood','median_price','moran_I']].to_string())", globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 16
import inspect
from esda.moran import Moran_Local
print(inspect.signature(Moran_Local.__init__))

# --- block 17 (wrapped in try/except)
try:
    exec("moran_local = Moran_Local(z, w, transformation='r', geoda_quads=True)\nprint('attrs:', [a for a in dir(moran_local) if not a.startswith('_')])\nneigh['moran_I'] = moran_local.I\nneigh['moran_z'] = moran_local.z_sim\nneigh['moran_p_sim'] = moran_local.p_sim\nprint(neigh[['neighbourhood','neighbourhood_group','median_price','moran_I','moran_z','moran_p_sim']].head(10).to_string())\nprint('\\nquads available:', hasattr(moran_local, 'quads'))", globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 18 (wrapped in try/except)
try:
    exec("print('Is shape:', moran_local.Is.shape, 'type:', type(moran_local.Is))\nprint('z_sim shape:', moran_local.z_sim.shape)\nprint('p_sim shape:', moran_local.p_sim.shape)\nprint(moran_local.Is[:5])\nprint('quads:', moran_local.quads[:5])\nneigh['moran_I'] = moran_local.Is\nneigh['moran_z'] = moran_local.z_sim\nneigh['moran_p_sim'] = moran_local.p_sim\nneigh['quad'] = moran_local.quads\nprint(neigh[['neighbourhood','median_price','moran_I','moran_z','moran_p_sim','quad']].head(12).to_string())", globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 19
neigh['moran_I'] = moran_local.Is
neigh['moran_z'] = moran_local.z_sim
neigh['moran_p_sim'] = moran_local.p_sim

# Quadrant classification (Geoda convention):
# HH: high value, high neighbours | LL: low value, low neighbours
# HL: high value, low neighbours  | LH: low value, high neighbours
high_z = neigh['moran_z'] > 0
neigh['category'] = np.where((neigh['moran_I'] > 0) & high_z, 'HH',
                  np.where((neigh['moran_I'] < 0) & high_z, 'HL',
                  np.where((neigh['moran_I'] < 0) & (~high_z), 'LL', 'LH')))
print(neigh['category'].value_counts())
print('\nquadrant counts from esda:', dict(zip(['HH','HL','LL','LH'], moran_local.quads)))

sig = neigh['moran_p_sim'] < 0.05
neigh['significant'] = sig
print('\nsignificant (p<0.05):', sig.sum())
print('\nHH significant (hot spots):')
print(neigh[(neigh['category']=='HH') & sig][['neighbourhood','median_price','moran_I','moran_z','moran_p_sim']].to_string())
print('\nLL significant (cold spots):')
print(neigh[(neigh['category']=='LL') & sig][['neighbourhood','median_price','moran_I','moran_z','moran_p_sim']].to_string())
print('\nHL significant:')
print(neigh[(neigh['category']=='HL') & sig][['neighbourhood','median_price','moran_I','moran_z','moran_p_sim']].to_string())
print('\nLH significant:')
print(neigh[(neigh['category']=='LH') & sig][['neighbourhood','median_price','moran_I','moran_z','moran_p_sim']].to_string())

# --- block 20
import inspect
src = inspect.getsource(type(moran_local).get_cluster_labels)
print(src)
src2 = inspect.getsource(type(moran_local).__init__)
# find geoda quads code
import re
for line in src2.split('\n'):
    if 'quad' in line.lower():
        print(line)

# --- block 21
import inspect
from esda.moran import _get_cluster_labels
print(inspect.getsource(_get_cluster_labels))
src3 = inspect.getsource(Moran_Local)
i = src3.find('def __quads')
print(src3[i:i+1200])

# --- block 22
# Compute quadrant directly from centered value and spatial lag (ground truth)
z_c = z - z.mean()
# spatial lag (row-standardized)
lag = np.asarray(w.sparse.dot(z_c)).flatten()

# My direct classification
my_q = np.where((z_c > 0) & (lag > 0), 'HH',
          np.where((z_c < 0) & (lag > 0), 'LH',
          np.where((z_c < 0) & (lag < 0), 'LL', 'HL')))

print('My quadrant counts:', dict(pd.Series(my_q).value_counts()))
print('esda q counts:', dict(pd.Series(moran_local.q).value_counts()))
print('esda cluster labels:', dict(pd.Series(moran_local.get_cluster_labels(0.05)).value_counts()))

# Cross check: does esda q map to my_q?
labels = {1:'High-High', 2:'Low-High', 3:'Low-Low', 4:'High-Low'}  # per _get_cluster_labels
q_map = dict(zip(moran_local.q, my_q))
print('\nesda q -> my_q mapping:')
for k,v in sorted(set(zip(moran_local.q.astype(int), my_q))):
    print(f'  q={k} -> {v}')

# --- block 23
# Assign category using direct classification (ground truth)
neigh['category'] = my_q
print(neigh['category'].value_counts())

sig = neigh['moran_p_sim'] < 0.05
neigh['significant'] = sig
print('significant total:', sig.sum())

for cat, name in [('HH','Hot spots (HH - high value, high neighbors)'),
                  ('LL','Cold spots (LL - low value, low neighbors)'),
                  ('HL','Outliers HL (high value, low neighbors)'),
                  ('LH','Outliers LH (low value, high neighbors)')]:
    sub = neigh[(neigh['category']==cat) & sig]
    print(f'\n{name}: {len(sub)} significant')
    if len(sub):
        print(sub[['neighbourhood','neighbourhood_group','median_price','moran_I','moran_p_sim']].to_string())

# --- block 24
# Full cold spots (LL significant)
ll_sig = neigh[(neigh['category']=='LL') & sig]
print('COLD SPOTS (LL, significant):', len(ll_sig))
print(ll_sig[['neighbourhood','neighbourhood_group','median_price','moran_I','moran_p_sim']].to_string())

# Save final results
neigh.to_file('pred_results/neigh_moran.geojson', driver='GeoJSON')
neigh.drop(columns=['geometry']).to_csv('pred_results/moran_local_results.csv', index=False)
print('\nSaved neighbourhood-level results.')

# Check projected CRS for nicer aspect
berlin_proj = neigh.to_crs('EPSG:32633')
print('\nProjected extent (32633):')
print(berlin_proj.total_bounds)

# --- block 25 (wrapped in try/except)
try:
    exec('import matplotlib\nmatplotlib.use(\'Agg\')\nimport matplotlib.pyplot as plt\nfrom matplotlib.lines import Line2D\nfrom matplotlib.patches import Patch\n\nproj = neigh.to_crs(\'EPSG:32633\')\nhh_sig = proj[(proj[\'category\']==\'HH\') & (proj.index.isin(neigh[(neigh[\'category\']==\'HH\') & neigh[\'significant\']].index))]\nll_sig_p = proj[(proj[\'category\']==\'LL\') & (proj.index.isin(neigh[(neigh[\'category\']==\'LL\') & neigh[\'significant\']].index))]\n\n# Build figure: main map + moran scatter\nfig = plt.figure(figsize=(16, 9))\ngs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.08)\nax = fig.add_subplot(gs[0])\nax_s = fig.add_subplot(gs[1])\n\n# Background: all neighbourhoods colored by median price\npm = proj.plot(column=\'median_price\', ax=ax, cmap=\'YlOrRd\', legend=False,\n               linewidth=0.3, edgecolor=\'white\', alpha=0.85)\n\n# Outline the whole of Berlin (exterior hull) in gray\nfrom shapely.ops import unary_union\nberlin_hull = unary_union(proj.geometry).envelope\nax.add_patch(plt.Polygon(np.array(berlin_hull.exterior.xy[0]), closed=True,\n             fill=False, edgecolor=\'gray\', lw=1.2, ls=\'--\', zorder=2))\n\n# Hot spots: thick red-orange edge\nhh_sig.plot(ax=ax, linewidth=2.2, edgecolor=\'red\', facecolor=\'none\', zorder=3, label=\'Hot spot (High-High, p<0.05)\')\n# Cold spots: thick blue edge\nll_sig_p.plot(ax=ax, linewidth=2.2, edgecolor=\'blue\', facecolor=\'none\', zorder=3, label=\'Cold spot (Low-Low, p<0.05)\')\n\n# colorbar for median price\nsm = plt.cm.ScalarMappable(cmap=\'YlOrRd\', norm=matplotlib.colors.Normalize(vmin=proj[\'median_price\'].min(), vmax=proj[\'median_price\'].max()))\nsm.set_array([])\ncb = fig.colorbar(sm, ax=ax, pad=0.08, aspect=20)\ncb.set_label(\'Median Airbnb price per neighbourhood group (€)\', fontsize=11)\n\nax.set_axis_off()\nax.set_title(\'Local Moran\\\'s I of Airbnb Prices in Berlin\\n\'\n             f\'Hot spots (HH) = {len(hh_sig)}   •   Cold spots (LL) = {len(ll_sig_p)}   •   Global Moran\\\'s I = {mi.I:.3f} (p = {mi.p_sim:.3f})\',\n             fontsize=13, pad=10)\nax.legend(loc=\'lower left\', fontsize=10, framealpha=0.9)\n\n# LISA scatter plot: y vs spatial lag\nax_s.scatter(lag, z, c=proj[\'median_price\'], cmap=\'YlOrRd\', s=28, alpha=0.75, edgecolors=\'k\', linewidths=0.3)\n# mark significant\nax_s.scatter(lag[sig.values], z[sig.values], facecolors=\'none\', edgecolors=\'black\', s=60, linewidths=1.0, zorder=4, label=\'Significant (p<0.05)\')\n# quadrants lines\nax_s.axhline(0, color=\'gray\', lw=0.8)\nax_s.axvline(0, color=\'gray\', lw=0.8)\nax_s.set_xlabel(\'Neighbouring value (spatial lag)\', fontsize=10)\nax_s.set_ylabel(\'Median price (centred)\', fontsize=10)\nax_s.set_title("LISA scatter plot (Local Moran\'s I)", fontsize=12)\nax_s.legend(fontsize=9)\nax_s.text(0.98, 0.97, \'HH\\n(hot)\', transform=ax_s.transAxes, ha=\'right\', va=\'top\', fontsize=10, color=\'red\', fontweight=\'bold\')\nax_s.text(0.98, 0.03, \'HL\', transform=ax_s.transAxes, ha=\'right\', va=\'bottom\', fontsize=10, color=\'green\')\nax_s.text(0.02, 0.03, \'LL\\n(cold)\', transform=ax_s.transAxes, ha=\'left\', va=\'bottom\', fontsize=10, color=\'blue\', fontweight=\'bold\')\nax_s.text(0.02, 0.97, \'LH\', transform=ax_s.transAxes, ha=\'left\', va=\'top\', fontsize=10, color=\'purple\')\n\nfig.suptitle(\'Spatial Autocorrelation of Airbnb Prices in Berlin\', fontsize=15, fontweight=\'bold\', y=0.98)\nfig.tight_layout(rect=[0, 0, 1, 0.95])\nfig.savefig(\'pred_results/moran_local.png\', dpi=150, bbox_inches=\'tight\')\nprint(\'saved figure\')', globals())
except Exception as _e:
    print('block failed:', repr(_e)[:200])

# --- block 26
fig = plt.figure(figsize=(16, 9))
gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1], wspace=0.08)
ax = fig.add_subplot(gs[0])
ax_s = fig.add_subplot(gs[1])

# Background: all neighbourhoods colored by median price
proj.plot(column='median_price', ax=ax, cmap='YlOrRd', legend=False,
          linewidth=0.3, edgecolor='white', alpha=0.85)

# Outline the whole of Berlin in gray
berlin_hull = unary_union(proj.geometry).envelope
ax.add_patch(plt.Polygon(np.array(berlin_hull.exterior.coords), closed=True,
             fill=False, edgecolor='gray', lw=1.2, ls='--', zorder=2))

# Hot spots: thick red edge
hh_sig.plot(ax=ax, linewidth=2.2, edgecolor='red', facecolor='none', zorder=3, label='Hot spot (High-High, p<0.05)')
# Cold spots: thick blue edge
ll_sig_p.plot(ax=ax, linewidth=2.2, edgecolor='blue', facecolor='none', zorder=3, label='Cold spot (Low-Low, p<0.05)')

sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=matplotlib.colors.Normalize(vmin=proj['median_price'].min(), vmax=proj['median_price'].max()))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, pad=0.08, aspect=20)
cb.set_label('Median Airbnb price per neighbourhood group (€)', fontsize=11)

ax.set_axis_off()
ax.set_title(f"Local Moran's I of Airbnb Prices in Berlin\n"
             f"Hot spots (HH) = {len(hh_sig)}   •   Cold spots (LL) = {len(ll_sig_p)}   •   Global Moran's I = {mi.I:.3f} (p = {mi.p_sim:.3f})",
             fontsize=13, pad=10)
ax.legend(loc='lower left', fontsize=10, framealpha=0.9)

# LISA scatter plot
ax_s.scatter(lag, z_c, c=proj['median_price'], cmap='YlOrRd', s=28, alpha=0.75, edgecolors='k', linewidths=0.3)
ax_s.scatter(lag[sig.values], z_c[sig.values], facecolors='none', edgecolors='black', s=60, linewidths=1.0, zorder=4, label='Significant (p<0.05)')
ax_s.axhline(0, color='gray', lw=0.8)
ax_s.axvline(0, color='gray', lw=0.8)
ax_s.set_xlabel('Neighbouring value (spatial lag, centred)', fontsize=10)
ax_s.set_ylabel('Median price (centred)', fontsize=10)
ax_s.set_title("LISA scatter plot (Local Moran's I)", fontsize=12)
ax_s.legend(fontsize=9, loc='center left')
ax_s.text(0.98, 0.97, 'HH (hot)', transform=ax_s.transAxes, ha='right', va='top', fontsize=10, color='red', fontweight='bold')
ax_s.text(0.98, 0.03, 'HL', transform=ax_s.transAxes, ha='right', va='bottom', fontsize=10, color='green')
ax_s.text(0.02, 0.03, 'LL (cold)', transform=ax_s.transAxes, ha='left', va='bottom', fontsize=10, color='blue', fontweight='bold')
ax_s.text(0.02, 0.97, 'LH', transform=ax_s.transAxes, ha='left', va='top', fontsize=10, color='purple')

fig.suptitle('Spatial Autocorrelation of Airbnb Prices in Berlin', fontsize=15, fontweight='bold', y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('pred_results/moran_local.png', dpi=150, bbox_inches='tight')
print('figure saved')

# --- block 27
import os
files = os.listdir('pred_results')
print('Files in pred_results:')
for f in sorted(files):
    print(f'  {f}  ({os.path.getsize(os.path.join("pred_results", f))/1024:.0f} KB)')

# Verify the figure
from PIL import Image
img = Image.open('pred_results/moran_local.png')
print('\nmoran_local.png size:', img.size)

# Final summary of the analysis
print('\n=== ANALYSIS SUMMARY ===')
print(f'Neighbourhoods analyzed: {len(neigh)}')
print(f'Global Moran\'s I: {mi.I:.4f} (p_sim={mi.p_sim}, z={mi.z_sim:.2f})')
print(f'\nSignificant local clusters (p<0.05):')
print(f'  HH (hot spots): {len(hh_sig)} - e.g. Mitte, Pankow, Charlottenburg (expensive areas)')
print(f'  LL (cold spots): {len(ll_sig_p)} - e.g. Reinickendorf, Spandau, Lichtenberg (cheaper areas)')
print(f'\nNeighbourhood group medians (€):')
print(group_median.sort_values(ascending=False).to_string())

# --- block 28
import numpy as np
arr = np.array(img.convert('RGB'))
print('Image shape:', arr.shape)
print('Unique-ish colors (sampled):', len(np.unique(arr.reshape(-1,3), axis=0)))
# fraction of non-white pixels
nonwhite = (arr < 240).any(axis=2).mean()
print(f'Fraction of non-white pixels: {nonwhite:.2%}')
# check red and blue edge presence
red = ((arr[:,:,0]>200)&(arr[:,:,1]<80)&(arr[:,:,2]<80)).sum()
blue = ((arr[:,:,2]>200)&(arr[:,:,0]<80)&(arr[:,:,1]<80)).sum()
print('Red edge pixels:', red, '| Blue edge pixels:', blue)
