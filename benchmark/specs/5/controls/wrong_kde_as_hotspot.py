# Wrong: "hot spot" = kernel density of cases at each hexagon centre (no Gi*, no significance testing).
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt, libpysal
from shapely.geometry import Polygon
from esda.getisord import G_Local
pts = gpd.read_file("dataset/High_Blood_Level_Results.shp").to_crs(32610)
zips = gpd.read_file("dataset/Sacramento_ZIP_Codes.shp").to_crs(32610)
s = 620.0
xmin, ymin, xmax, ymax = zips.total_bounds
cells = []
for i, x in enumerate(np.arange(xmin, xmax + 2 * s, 1.5 * s)):
    for y in np.arange(ymin - (np.sqrt(3) * s / 2 if i % 2 else 0), ymax + 2 * s, np.sqrt(3) * s):
        cells.append(Polygon([(x + s * np.cos(a), y + s * np.sin(a)) for a in np.radians(range(0, 360, 60))]))
hexes = gpd.GeoDataFrame(geometry=cells, crs=32610)
hexes = hexes[hexes.intersects(zips.union_all())].reset_index(drop=True)
hexes["n"] = gpd.sjoin(pts, hexes, predicate="within").groupby("index_right").size().reindex(hexes.index, fill_value=0)

from scipy.stats import gaussian_kde
kde = gaussian_kde(np.vstack([pts.geometry.x, pts.geometry.y]), bw_method=0.15)
hexes["hotspot"] = kde(np.vstack([hexes.centroid.x, hexes.centroid.y]))
hexes["hot"] = hexes["hotspot"] > hexes["hotspot"].quantile(0.9)
ax = hexes.plot(column="hotspot", cmap="Reds", legend=True, figsize=(10, 10))
hexes[hexes.hot].boundary.plot(ax=ax, color="k")
plt.savefig("pred_results/transit_access.png")
