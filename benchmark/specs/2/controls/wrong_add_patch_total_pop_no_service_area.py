# Wrong: hand-drawn patches (ax.add_patch), but total population is drawn instead of density, and
# the service area is not drawn (only a bounding box of it).
import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon as MplPolygon
import shapely
from shapely.geometry import MultiPolygon, box

bus = gpd.read_file('dataset/BusServiceArea.geojson').to_crs(4326)
dem = gpd.read_file('dataset/HamiltonDemographics.geojson').to_crs(4326)
served = dem[dem.intersects(bus.geometry.union_all())].copy()
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, (col, cm) in zip(axes, [('populationtotals_TOTPOP_CY', 'YlOrRd'), ('households_ACSHHBPOV', 'Purples'), ('AtRisk_ACSOVEH0', 'Blues')]):
    cmap = plt.get_cmap(cm)
    norm = mcolors.Normalize(served[col].min(), served[col].max())
    for geom, val in zip(served.geometry, served[col]):
        for part in (shapely.get_parts(geom) if isinstance(geom, MultiPolygon) else [geom]):
            x, y = part.exterior.xy
            ax.add_patch(MplPolygon(np.column_stack([x, y]), closed=True, facecolor=cmap(norm(val)),
                                    alpha=0.7, edgecolor='#8a8a8a', linewidth=0.5, zorder=2))
    bb = box(*bus.total_bounds)
    x, y = bb.exterior.xy
    ax.add_patch(MplPolygon(np.column_stack([x, y]), closed=True, facecolor='none', edgecolor='red', zorder=3))
    b = served.total_bounds
    ax.set_xlim(b[0], b[2]); ax.set_ylim(b[1], b[3])
fig.savefig('pred_results/transit_access.png', dpi=100)
