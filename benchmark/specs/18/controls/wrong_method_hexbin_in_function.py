# Wrong: the task prescribes a "quadtree format map"; this aggregates by a regular grid / hexagons, not quadtree cells.
# Hexagonal aggregation (mean of null flag per hexagon) done inside a function, result only in the figure.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
def make_map():
    t = gpd.read_file('dataset/street_trees_sample.geojson').to_crs(32610)
    city = gpd.read_file('dataset/sfo_boroughs.geojson').to_crs(32610)
    flag = t['Species'].isnull().to_numpy().astype(float)
    fig, ax = plt.subplots(figsize=(8, 8))
    hb = ax.hexbin(t.geometry.x, t.geometry.y, C=flag * 100, reduce_C_function=np.mean, gridsize=25, mincnt=10, cmap='Reds')
    city.boundary.plot(ax=ax, color='k')
    fig.colorbar(hb, label='% NULL species')
    fig.savefig('pred_results/trees_count_vis.png')
make_map()
