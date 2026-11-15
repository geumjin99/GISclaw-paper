# Correct: every block group and the service area drawn by hand with
# ax.add_patch(matplotlib Polygon), facecolor = cmap(norm(value)); density = pop/area,
# poverty min-max scaled, no-vehicle share.
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
import shapely
from shapely.geometry import MultiPolygon

bus = gpd.read_file('dataset/BusServiceArea.geojson').to_crs(4326)
dem = gpd.read_file('dataset/HamiltonDemographics.geojson')
dem_m = dem.to_crs(32616)
dem['pop_density'] = dem['populationtotals_TOTPOP_CY'] / (dem_m.geometry.area / 1e6)
p = dem['households_ACSHHBPOV'].astype(float)
dem['poverty_value'] = (p - p.min()) / (p.max() - p.min())
dem['no_vehicle_value'] = dem['AtRisk_ACSOVEH0'] / dem['populationtotals_TOTPOP_CY'].replace(0, np.nan)
served = dem.to_crs(4326)
served = served[served.intersects(bus.geometry.union_all())].copy()
served['no_vehicle_value'] = served['no_vehicle_value'].fillna(0)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (col, cm) in zip(axes, [('pop_density', 'YlOrRd'), ('poverty_value', 'Purples'), ('no_vehicle_value', 'Blues')]):
    cmap = plt.get_cmap(cm)
    norm = mcolors.Normalize(served[col].min(), served[col].max())
    for geom, val in zip(served.geometry, served[col]):
        for part in (shapely.get_parts(geom) if isinstance(geom, MultiPolygon) else [geom]):
            x, y = part.exterior.xy
            ax.add_patch(MplPolygon(np.column_stack([x, y]), closed=True, facecolor=cmap(norm(val)),
                                    alpha=0.7, edgecolor='#8a8a8a', linewidth=0.5, zorder=2))
    g = bus.geometry.union_all()
    for part in (shapely.get_parts(g) if isinstance(g, MultiPolygon) else [g]):
        x, y = part.exterior.xy
        ax.add_patch(MplPolygon(np.column_stack([x, y]), closed=True, facecolor='none',
                                edgecolor='red', linewidth=2, zorder=3))
    b = served.total_bounds
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3]); ax.set_title(col)
fig.savefig('pred_results/transit_access.png', dpi=100)
