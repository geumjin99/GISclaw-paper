# Wrong: risky area right, protected and deforested drawn, but
# the existing roads are drawn (as one NaN-separated array with ax.plot) only where they meet the buffer.
import numpy as np, geopandas as gpd, shapely, matplotlib.pyplot as plt
C = 32720
roads = gpd.read_file("dataset/roads.geojson").to_crs(C)
deforested = gpd.read_file("dataset/deforestedArea.geojson").to_crs(C)
deforested["geometry"] = deforested.geometry.make_valid()
protected = gpd.read_file("dataset/protectedForest.geojson").to_crs(C)
planned = gpd.read_file("dataset/planned_road.geojson").to_crs(C)
buf = planned.geometry.union_all().buffer(5500)
risk = gpd.GeoDataFrame(geometry=[buf.difference(deforested.geometry.union_all())], crs=C)


def seg_array(geoms):
    rows = []
    for g in geoms:
        for ln in (g.geoms if g.geom_type == "MultiLineString" else [g]):
            rows.extend(np.asarray(ln.coords)); rows.append([np.nan, np.nan])
    return np.array(rows)


near = roads[roads.intersects(buf.buffer(20000))]
arr = seg_array(near.geometry)
fig, ax = plt.subplots(figsize=(10, 10))
protected.plot(ax=ax, color="#a6d96a"); deforested.plot(ax=ax, color="#d8b365", alpha=0.6)
ax.plot(arr[:, 0], arr[:, 1], color="grey", lw=0.3)
risk.plot(ax=ax, color="red"); planned.plot(ax=ax, color="blue")
plt.savefig("pred_results/predictedRiskyArea.png")
