# Wrong: three quantile layers drawn with explicit colour lists (plot(color=[...]), no column=), but the
# first layer is TOTAL population, not population density.
import numpy as np, pandas as pd, geopandas as gpd, mapclassify, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
bus = gpd.read_file("dataset/BusServiceArea.geojson").to_crs(32616)
demo = gpd.read_file("dataset/HamiltonDemographics.geojson").to_crs(32616)
demo['total_pop'] = pd.to_numeric(demo['populationtotals_TOTPOP_CY'], errors='coerce')
demo['poverty'] = pd.to_numeric(demo['households_ACSHHBPOV'], errors='coerce')
demo['no_vehicle'] = pd.to_numeric(demo['AtRisk_ACSOVEH0'], errors='coerce')
served = gpd.overlay(demo, bus[['geometry']], how='intersection', keep_geom_type=True)
fig, ax = plt.subplots(figsize=(12, 12))
demo.plot(ax=ax, color='#e6e6e6', edgecolor='#999999', linewidth=0.3)
for col, cmap in [('total_pop', 'Blues'), ('poverty', 'Reds'), ('no_vehicle', 'Greens')]:
    q = mapclassify.Quantiles(served[col], k=5)
    k = len(q.bins); cm = plt.get_cmap(cmap)
    colors = [cm(0.25 + 0.75 * i / max(k - 1, 1)) for i in range(k)]
    served.plot(ax=ax, color=[colors[i] for i in q.yb], alpha=0.4, edgecolor='none')
bus.boundary.plot(ax=ax, color='black', linewidth=2)
fig.savefig('pred_results/transit_access.png', dpi=120, bbox_inches='tight')
