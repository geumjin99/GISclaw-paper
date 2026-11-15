"""Reference solution for task 9: deforestation rate within 5.5 km of roads, Rondonia.

percentage_deforestation = 100 * area(deforested AND road buffer) / area(road buffer), where the
road buffer is the dissolved 5.5 km buffer of all 27 662 road segments (any status).
Areas in South America Albers Equal Area (ESRI:102033).

The deforested layer is one invalid multipolygon (8 257 parts, 0.5 M vertices) and the dissolved
road buffer covers ~165 000 km2, so a single global overlay takes very long. The overlay is done
exactly but tile by tile (24 x 24 tiles; clip -> make_valid -> union -> intersect, areas summed),
which gives the same areas as one global overlay.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

CRS_M = "ESRI:102033"
DIST = 5500.0
N_TILES = 24


def main():
    roads = gpd.read_file("dataset/roads.geojson").to_crs(CRS_M)
    deforested = gpd.read_file("dataset/deforestedArea.geojson").to_crs(CRS_M)
    bufs = shapely.buffer(roads.geometry.values, DIST, quad_segs=8)
    parts = np.array(shapely.get_parts(deforested.geometry.values))
    btree, ptree = shapely.STRtree(bufs), shapely.STRtree(parts)
    xmin, ymin, xmax, ymax = shapely.total_bounds(bufs)
    xs = np.linspace(xmin, xmax, N_TILES + 1)
    ys = np.linspace(ymin, ymax, N_TILES + 1)
    buffer_area = deforested_in_buffer = 0.0
    for i in range(N_TILES):
        for j in range(N_TILES):
            rect = (xs[i], ys[j], xs[i + 1], ys[j + 1])
            bi = btree.query(shapely.box(*rect))
            if not len(bi):
                continue
            b = shapely.union_all(shapely.clip_by_rect(bufs[bi], *rect))
            buffer_area += b.area
            pi = ptree.query(shapely.box(*rect))
            if not len(pi):
                continue
            d = shapely.make_valid(shapely.union_all(shapely.make_valid(shapely.clip_by_rect(parts[pi], *rect))))
            deforested_in_buffer += shapely.intersection(b, d).area
    percentage = 100.0 * deforested_in_buffer / buffer_area
    print(f"road buffer {buffer_area / 1e6:.0f} km2, deforested inside {deforested_in_buffer / 1e6:.0f} km2 "
          f"-> {percentage:.2f} %")
    df = pd.DataFrame({"percentage_deforestation": [round(percentage, 4)]})
    df.to_csv("pred_results/deforestation_rate.csv", index=False)
    return df, buffer_area, deforested_in_buffer


df, buffer_area, deforested_in_buffer = main()
