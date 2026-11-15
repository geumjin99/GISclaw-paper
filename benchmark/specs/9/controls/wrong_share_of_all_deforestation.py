# Wrong: divides the deforested area inside the buffer by the total deforested area (share of
# deforestation near roads) instead of by the buffer area.
import numpy as np, pandas as pd, geopandas as gpd, shapely
def tiled(bufs, parts, n=24, op="intersection"):
    bt, pt = shapely.STRtree(bufs), shapely.STRtree(parts)
    x0, y0, x1, y1 = shapely.total_bounds(bufs)
    xs, ys = np.linspace(x0, x1, n + 1), np.linspace(y0, y1, n + 1)
    A = B = 0.0
    for i in range(n):
        for j in range(n):
            r = (xs[i], ys[j], xs[i + 1], ys[j + 1])
            bi = bt.query(shapely.box(*r))
            if not len(bi):
                continue
            b = shapely.union_all(shapely.clip_by_rect(bufs[bi], *r)); A += b.area
            pi = pt.query(shapely.box(*r))
            d = shapely.make_valid(shapely.union_all(shapely.make_valid(shapely.clip_by_rect(parts[pi], *r)))) if len(pi) else shapely.Polygon()
            B += (shapely.intersection(b, d) if op == "intersection" else shapely.difference(b, d)).area
    return A, B
roads = gpd.read_file('dataset/roads.geojson').to_crs('ESRI:102033')
defo = gpd.read_file('dataset/deforestedArea.geojson').to_crs('ESRI:102033')
A, B = tiled(shapely.buffer(roads.geometry.values, 5500), np.array(shapely.get_parts(defo.geometry.values)))
total = sum(shapely.make_valid(p).area for p in shapely.get_parts(defo.geometry.values))
pd.DataFrame({'percentage_deforestation': [100 * B / total]}).to_csv('pred_results/deforestation_rate.csv', index=False)
