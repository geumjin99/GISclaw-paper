# Correct: H3 resolution-8 hexagons (~0.74 km2), case counts, Gi* with kNN-6 weights, 99 perms,
# results classified into confidence bins; all inside a function.
import numpy as np, geopandas as gpd, h3, libpysal, matplotlib.pyplot as plt
from shapely.geometry import Polygon
from esda.getisord import G_Local

def run():
    cases = gpd.read_file('dataset/High_Blood_Level_Results.shp').to_crs(4326)
    zips = gpd.read_file('dataset/Sacramento_ZIP_Codes.shp').to_crs(4326)
    area = zips.union_all()
    ids = set()
    for poly in getattr(area, 'geoms', [area]):
        ids |= set(h3.geo_to_cells(h3.LatLngPoly([(y, x) for x, y in poly.exterior.coords]), 8))
    ids |= {h3.latlng_to_cell(y, x, 8) for x, y in zip(cases.geometry.x, cases.geometry.y)}
    ids = sorted(ids)
    grid = gpd.GeoDataFrame({'h3': ids}, geometry=[Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in ids], crs=4326)
    cell_of_case = [h3.latlng_to_cell(y, x, 8) for x, y in zip(cases.geometry.x, cases.geometry.y)]
    grid['n_cases'] = grid['h3'].map(gpd.pd.Series(cell_of_case).value_counts()).fillna(0)
    g_m = grid.to_crs(32610)
    w = libpysal.weights.KNN.from_dataframe(g_m, k=6)
    gi = G_Local(grid['n_cases'].values, w, star=True, permutations=99, seed=1)
    conf = np.where(gi.p_sim < 0.01, 3, np.where(gi.p_sim < 0.05, 2, np.where(gi.p_sim < 0.1, 1, 0)))
    grid['hs_bin'] = np.sign(gi.Zs) * conf
    fig, ax = plt.subplots(figsize=(10, 10))
    grid.plot(column='hs_bin', cmap='coolwarm', categorical=True, legend=True, ax=ax, linewidth=0)
    zips.boundary.plot(ax=ax, color='k', linewidth=0.5)
    ax.set_title('Blood lead hot spots (Gi*, H3 res 8)')
    fig.savefig('pred_results/transit_access.png', dpi=120)
run()
