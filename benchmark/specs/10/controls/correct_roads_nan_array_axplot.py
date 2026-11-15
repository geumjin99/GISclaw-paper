# Correct: the existing roads are drawn via a helper that turns the layer into one NaN-separated coordinate
# array, then ax.plot. The steps run in order; steps that may raise are wrapped in try/except.
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)


roads_raw = gpd.read_file('dataset/roads.geojson')
print(f"Rows: {len(roads_raw)}")
print(f"CRS: {roads_raw.crs.to_string() if roads_raw.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in roads_raw.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in roads_raw.total_bounds]}")
_cols = [c for c in roads_raw.columns if c != roads_raw.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(roads_raw[_cols].head(3).to_string())

planned_raw = gpd.read_file('dataset/planned_road.geojson')
print(f"Rows: {len(planned_raw)}")
print(f"CRS: {planned_raw.crs.to_string() if planned_raw.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in planned_raw.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in planned_raw.total_bounds]}")
_cols = [c for c in planned_raw.columns if c != planned_raw.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(planned_raw[_cols].head(3).to_string())

deforested_raw = gpd.read_file('dataset/deforestedArea.geojson')
print(f"Rows: {len(deforested_raw)}")
print(f"CRS: {deforested_raw.crs.to_string() if deforested_raw.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in deforested_raw.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in deforested_raw.total_bounds]}")
_cols = [c for c in deforested_raw.columns if c != deforested_raw.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(deforested_raw[_cols].head(3).to_string())

protected_raw = gpd.read_file('dataset/protectedForest.geojson')
print(f"Rows: {len(protected_raw)}")
print(f"CRS: {protected_raw.crs.to_string() if protected_raw.crs is not None else None}")
print(f"Geometry types: { {k: int(c) for k, c in protected_raw.geom_type.value_counts().items()} }")
print(f"Bounds (minx, miny, maxx, maxy): {[round(float(_b), 4) for _b in protected_raw.total_bounds]}")
_cols = [c for c in protected_raw.columns if c != protected_raw.geometry.name]
print(f"Columns ({len(_cols)}): {_cols}")
import pandas as _pd
with _pd.option_context('display.max_columns', 30, 'display.width', 200, 'display.max_colwidth', 40):
    print(protected_raw[_cols].head(3).to_string())

try:  # errors in this block are tolerated

    import geopandas as gpd
    import shapely

    print("planned_raw CRS:", planned_raw.crs)
    print("roads invalid:", (~roads_raw.geometry.is_valid()).sum(), "of", len(roads_raw))
    print("deforested invalid:", (~deforested_raw.geometry.is_valid()).sum(), "of", len(deforested_raw))
    print("protected invalid:", (~protected_raw.geometry.is_valid()).sum(), "of", len(protected_raw))

    # vertex counts (for boundary hull feasibility)
    n_roads_pts = sum(len(shapely.get_coordinates(g)) for g in roads_raw.geometry)
    n_def_pts = len(shapely.get_coordinates(deforested_raw.geometry.iloc[0]))
    n_prot_pts = sum(len(shapely.get_coordinates(g)) for g in protected_raw.geometry)
    print("road vertices:", n_roads_pts, "| deforested vertices:", n_def_pts, "| protected vertices:", n_prot_pts)
except Exception:
    pass


import shapely

inv_roads = (~shapely.is_valid(roads_raw.geometry)).sum()
inv_def = (~shapely.is_valid(deforested_raw.geometry)).sum()
inv_prot = (~shapely.is_valid(protected_raw.geometry)).sum()
print("invalid -> roads:", inv_roads, "deforested:", inv_def, "protected:", inv_prot)

n_roads_pts = sum(len(shapely.get_coordinates(g)) for g in roads_raw.geometry)
n_def_pts = len(shapely.get_coordinates(deforested_raw.geometry.iloc[0]))
n_prot_pts = sum(len(shapely.get_coordinates(g)) for g in protected_raw.geometry)
print("road vertices:", n_roads_pts, "| deforested vertices:", n_def_pts, "| protected vertices:", n_prot_pts)

# check planned road bounds in its native CRS (sanity vs state bounds)
print("planned bounds:", planned_raw.total_bounds)

try:  # errors in this block are tolerated

    import shapely
    from shapely.geometry import MultiPoint
    import pyproj

    # 1) target CRS: the projected equal-area CRS of planned_road (resolves fine)
    target_crs = planned_raw.crs
    crs_resolved = pyproj.CRS.from_user_input("ESRI:102033").is_valid
    print("target_crs:", target_crs.name if hasattr(target_crs, 'name') else target_crs, "| resolved:", crs_resolved)
    if not crs_resolved:
        target_crs = pyproj.CRS.from_epsg(5880)
        print("FALLBACK to EPSG:5880")

    # 2) repair invalid geometries
    deforested_raw["geometry"] = shapely.make_valid(deforested_raw.geometry)
    protected_raw["geometry"] = shapely.make_valid(protected_raw.geometry)
    planned_raw["geometry"] = shapely.make_valid(planned_raw.geometry)
    print("after repair invalid -> deforested:", (~shapely.is_valid(deforested_raw.geometry)).sum(),
          "| protected:", (~shapely.is_valid(protected_raw.geometry)).sum(),
          "| planned:", (~shapely.is_valid(planned_raw.geometry)).sum())

    # 3) approximate state boundary: convex hull of all vertices of roads + deforested + protected (in 4326)
    import numpy as np
    pts = np.vstack([
        shapely.get_coordinates(roads_raw.geometry.values),
        shapely.get_coordinates(deforested_raw.geometry.values),
        shapely.get_coordinates(protected_raw.geometry.values),
    ])
    print("hull input points:", pts.shape[0])
    state_hull_4326 = shapely.geometry.MultiPoint(pts).convex_hull
    print("hull bounds (4326):", state_hull_4326.bounds)

    # reproject boundary to target CRS to report area in m2 (equal-area CRS -> true area)
    state_boundary = gpd.GeoDataFrame(geometry=[state_hull_4326], crs=4326).to_crs(target_crs)
    print("state_boundary rows:", len(state_boundary), "CRS:", state_boundary.crs.to_string()[:60])
    print("boundary area (km2):", state_boundary.geometry.area.iloc[0] / 1e6)
except Exception:
    pass

try:  # errors in this block are tolerated

    import shapely
    from shapely.geometry import MultiPoint
    import numpy as np

    # 1) target CRS check (resolvable? projected equal-area?)
    crs_resolved = target_crs.is_projected() and "Albers" in target_crs.name
    print("target_crs:", target_crs.name, "| resolved:", crs_resolved)
    if not crs_resolved:
        target_crs = pyproj.CRS.from_epsg(5880)
        print("FALLBACK to EPSG:5880")

    # 2) repair invalid geometries
    deforested_raw["geometry"] = shapely.make_valid(deforested_raw.geometry)
    protected_raw["geometry"] = shapely.make_valid(protected_raw.geometry)
    planned_raw["geometry"] = shapely.make_valid(planned_raw.geometry)
    print("after repair invalid -> deforested:", (~shapely.is_valid(deforested_raw.geometry)).sum(),
          "| protected:", (~shapely.is_valid(protected_raw.geometry)).sum(),
          "| planned:", (~shapely.is_valid(planned_raw.geometry)).sum())

    # 3) approximate state boundary: convex hull of all vertices of roads + deforested + protected (in 4326)
    pts = np.vstack([
        shapely.get_coordinates(roads_raw.geometry.values),
        shapely.get_coordinates(deforested_raw.geometry.values),
        shapely.get_coordinates(protected_raw.geometry.values),
    ])
    print("hull input points:", pts.shape[0])
    state_hull_4326 = shapely.geometry.MultiPoint(pts).convex_hull
    print("hull bounds (4326):", state_hull_4326.bounds)

    state_boundary = gpd.GeoDataFrame(geometry=[state_hull_4326], crs=4326).to_crs(target_crs)
    print("state_boundary rows:", len(state_boundary), "CRS:", target_crs.name)
    print("boundary area (km2):", state_boundary.geometry.area.iloc[0] / 1e6)
except Exception:
    pass


import shapely
from shapely.geometry import MultiPoint
import numpy as np

# 1) target CRS check
crs_resolved = target_crs.is_projected and "Albers" in target_crs.name
print("target_crs:", target_crs.name, "| resolved:", crs_resolved)
if not crs_resolved:
    target_crs = pyproj.CRS.from_epsg(5880)
    print("FALLBACK to EPSG:5880")

# 2) repair invalid geometries
deforested_raw["geometry"] = shapely.make_valid(deforested_raw.geometry)
protected_raw["geometry"] = shapely.make_valid(protected_raw.geometry)
planned_raw["geometry"] = shapely.make_valid(planned_raw.geometry)
print("after repair invalid -> deforested:", (~shapely.is_valid(deforested_raw.geometry)).sum(),
      "| protected:", (~shapely.is_valid(protected_raw.geometry)).sum(),
      "| planned:", (~shapely.is_valid(planned_raw.geometry)).sum())

# 3) approximate state boundary: convex hull of all vertices of roads + deforested + protected (in 4326)
pts = np.vstack([
    shapely.get_coordinates(roads_raw.geometry.values),
    shapely.get_coordinates(deforested_raw.geometry.values),
    shapely.get_coordinates(protected_raw.geometry.values),
])
print("hull input points:", pts.shape[0])
state_hull_4326 = MultiPoint(pts).convex_hull
print("hull bounds (4326):", state_hull_4326.bounds)

state_boundary = gpd.GeoDataFrame(geometry=[state_hull_4326], crs=4326).to_crs(target_crs)
print("state_boundary rows:", len(state_boundary), "CRS:", target_crs.name)
print("boundary area (km2):", state_boundary.geometry.area.iloc[0] / 1e6)

try:  # errors in this block are tolerated

    import time

    t0 = time.time()
    deforested_area = deforested_raw.to_crs(target_crs).clip(state_boundary)
    print("deforested_area:", len(deforested_area), "features, time", round(time.time()-t0,1), "s")

    t0 = time.time()
    protected_forest = protected_raw.to_crs(target_crs).clip(state_boundary, keep_geometry_type=True)
    protected_forest = protected_forest[~protected_forest.geometry.is_empty]
    print("protected_forest:", len(protected_forest), "features (was 52), time", round(time.time()-t0,1), "s")

    t0 = time.time()
    planned_road = planned_raw.to_crs(target_crs)
    print("planned_road:", len(planned_road), "features, length m:", planned_road.geometry.length.iloc[0], "time", round(time.time()-t0,1), "s")
    print("planned_road inside boundary:", bool(planned_road.geometry.iloc[0].within(state_boundary.geometry.iloc[0])))
except Exception:
    pass

try:  # errors in this block are tolerated

    import time

    t0 = time.time()
    roads = roads_raw.to_crs(target_crs).clip(state_boundary)
    roads = roads[~roads.geometry.is_empty]
    print("roads:", len(roads), "features (was 27662), time", round(time.time()-t0,1), "s")
except Exception:
    pass

try:  # errors in this block are tolerated

    import time
    import geopandas as gpd
    import shapely

    state_geom = state_boundary.geometry.iloc[0]  # single polygon in target CRS

    def clip_to_state(gdf):
        t0 = time.time()
        proj = gdf.to_crs(target_crs)
        geoms = [g.intersection(state_geom) for g in proj.geometry.values]
        out = gpd.GeoDataFrame(proj[proj.columns.drop("geometry")],
                                geometry=gpd.GeoSeries(geoms, crs=target_crs),
                                index=proj.index)
        out = out[~out.geometry.is_empty]
        print(f"  {gdf.name}: {len(out)}/{len(proj)} features after clip, {round(time.time()-t0,1)} s")
        return out

    deforested_area = clip_to_state(deforested_raw.rename("deforested"))
    protected_forest = clip_to_state(protected_raw.rename("protected"))
    print("deforested area km2:", deforested_area.geometry.area.sum()/1e6)
    print("protected area km2:", protected_forest.geometry.area.sum()/1e6)
except Exception:
    pass


import time
import geopandas as gpd
import shapely

state_geom = state_boundary.geometry.iloc[0]  # single polygon in target CRS

def clip_to_state(gdf, label):
    t0 = time.time()
    proj = gdf.to_crs(target_crs)
    geoms = [g.intersection(state_geom) for g in proj.geometry.values]
    out = gpd.GeoDataFrame(proj[proj.columns.drop("geometry")],
                           geometry=gpd.GeoSeries(geoms, crs=target_crs),
                           index=proj.index)
    out = out[~out.geometry.is_empty]
    print(f"  {label}: {len(out)}/{len(proj)} features after clip, {round(time.time()-t0,1)} s")
    return out

deforested_area = clip_to_state(deforested_raw, "deforested")
protected_forest = clip_to_state(protected_raw, "protected")
print("deforested area km2:", deforested_area.geometry.area.sum()/1e6)
print("protected area km2:", protected_forest.geometry.area.sum()/1e6)


import time
import geopandas as gpd

# planned road (small, reproject; keep whole line then verify inside boundary)
t0 = time.time()
planned_road = planned_raw.to_crs(target_crs)
inb = planned_road.geometry.iloc[0].within(state_geom)
print("planned_road:", len(planned_road), "features | fully inside boundary:", bool(inb),
      "| length km:", round(planned_road.geometry.length.iloc[0]/1e3, 1), f"| {round(time.time()-t0,1)} s")

# roads (large)
t0 = time.time()
proj = roads_raw.to_crs(target_crs)
geoms = [g.intersection(state_geom) for g in proj.geometry.values]
roads = gpd.GeoDataFrame(proj[proj.columns.drop("geometry")],
                         geometry=gpd.GeoSeries(geoms, crs=target_crs),
                         index=proj.index)
roads = roads[~roads.geometry.is_empty]
print("roads:", len(roads), "features after clip (was 27662),", round(time.time()-t0,1), "s")
print("roads CRS:", roads.crs.name if roads.crs else None)


import shapely

for name, gdf in [("roads", roads), ("planned_road", planned_road),
                  ("deforested_area", deforested_area), ("protected_forest", protected_forest),
                  ("state_boundary", state_boundary)]:
    inv = (~shapely.is_valid(gdf.geometry)).sum()
    empty = gdf.geometry.is_empty.sum()
    print(f"{name:16s} rows={len(gdf):6d} crs={gdf.crs.name if gdf.crs is not None else None} "
          f"invalid={inv} empty={empty} bounds=({gdf.total_bounds[0]:.0f},{gdf.total_bounds[1]:.0f},"
          f"{gdf.total_bounds[2]:.0f},{gdf.total_bounds[3]:.0f})")

print()
print("target_crs:", target_crs.name, "| equal-area:", "Albers" in target_crs.name)
print("approx. Rondônia boundary area (km2):", round(state_boundary.geometry.area.iloc[0]/1e6, 1))
print("deforested_area (km2):", round(deforested_area.geometry.area.sum()/1e6, 1))
print("protected_forest (km2):", round(protected_forest.geometry.area.sum()/1e6, 1))
print("planned_road length (km):", round(planned_road.geometry.length.sum()/1e3, 1))

try:  # errors in this block are tolerated

    import numpy as np
    import geopandas as gpd
    from shapely.validation import make_valid

    # --- Create the 5.5 km buffer around the planned road in the projected CRS (metres) ---
    planned_buffer_raw = planned_road.copy()
    planned_buffer_raw["geometry"] = planned_road.geometry.buffer(5500)

    # Validate buffer geometry
    n_invalid = planned_buffer_raw.geometry.is_valid.eq(False).sum()
    print("Invalid buffer geometries:", n_invalid)
    if n_invalid > 0:
        planned_buffer_raw["geometry"] = make_valid(planned_buffer_raw["geometry"])
        print("After make_valid, invalid:", planned_buffer_raw.geometry.is_valid.eq(False).sum())

    # Reproject is not needed; buffer is already in the projected CRS (ESRI:102033, metres)
    print("Planned buffer CRS:", planned_buffer_raw.crs)
    print("Buffer geometry types:", planned_buffer_raw.geometry.geom_type.value_counts().to_dict())

    # Planned road length (m) and km
    planned_length_m = planned_road.geometry.length.iloc[0]
    print(f"Planned road length: {planned_length_m:.1f} m = {planned_length_m/1000:.2f} km")

    # --- Intersect buffer with approximate Rondônia boundary ---
    state_boundary_geom = state_boundary.geometry.iloc[0]
    print("Boundary valid:", state_boundary_geom.is_valid)

    planned_buffer = gpd.GeoDataFrame(
        planned_buffer_raw.drop(columns=["geometry"]),
        geometry=planned_buffer_raw.geometry,
        crs=planned_road.crs,
    )

    # Intersection with the boundary
    intersection = planned_buffer.intersection(state_boundary_geom)
    print("Intersection result type:", intersection.geom_type)
    print("Intersection valid:", intersection.is_valid)
    print("Intersection area (m2):", intersection.area)

    # --- Calculate the total buffer area inside the boundary ---
    buffer_area_m2 = intersection.area
    buffer_area_km2 = buffer_area_m2 / 1e6
    print(f"\nTotal buffer area inside the boundary: {buffer_area_km2:.2f} km2")

    # Sanity check: buffer area should be roughly 2 * length * 5500 for a long thin line
    expected_area_m2 = 2 * planned_length_m * 5500
    print(f"Expected approximate area (no overlap/cap): {expected_area_m2/1e6:.1f} km2")

    # Store the planned_buffer as a GeoDataFrame with the intersected geometry
    planned_buffer = gpd.GeoDataFrame(
        {"Name": ["BR 421"], "Status": ["Proposed"]},
        geometry=intersection,
        crs=planned_road.crs,
    )
    planned_buffer["area_km2"] = buffer_area_km2
    print("\nplanned_buffer (intersected with boundary):")
    print(planned_buffer)
except Exception:
    pass


# Re-derive clean scalars and verify consistency
planned_length_m = float(planned_road.geometry.length.iloc[0])
total_buffer_m2 = float(planned_buffer_raw.geometry.area.iloc[0])
inside_m2 = float(intersection.area.iloc[0])

print(f"Planned road length: {planned_length_m/1000:.2f} km")
print(f"Total buffer area (5.5 km): {total_buffer_m2/1e6:.2f} km2")
print(f"Buffer area inside Rondônia boundary: {inside_m2/1e6:.2f} km2")
print(f"Buffer area outside boundary: {(total_buffer_m2-inside_m2)/1e6:.2f} km2")

# Theory check: expected ≈ 2*L*d + pi*d^2
import math
expected = 2*planned_length_m*5500 + math.pi*5500**2
print(f"Theoretical buffer area (caps included): {expected/1e6:.2f} km2")

buffer_area_km2 = inside_m2/1e6
print(f"\nbuffer_area_km2 = {buffer_area_km2:.2f}")

# Confirm planned_buffer GeoDataFrame holds the intersected (boundary-clipped) geometry
print("\nplanned_buffer:")
print(planned_buffer)
print("planned_buffer CRS:", planned_buffer.crs)
print("geometry valid:", planned_buffer.geometry.is_valid.all())

try:  # errors in this block are tolerated

    print(planned_buffer.columns.tolist())
    print("area_km2 value:", float(planned_buffer["area_km2"].iloc[0]))
    print("buffer_area_km2 variable:", float(buffer_area_km2))
    print("geom type:", planned_buffer.geometry.geom_type.iloc[0])
    print("bounds:", planned_buffer.total_bounds)
    # Cross-check vs planned road extent: buffer should extend ~5.5km beyond the road
    rb = planned_road.total_bounds
    bb = planned_buffer.total_bounds
    print("planned road bounds:", rb)
    print("buffer extends beyond road:", bb[0] <= rb[0], bb[1] <= rb[1], bb[2] >= rb[2], bb[3] >= rb[3])
except Exception:
    pass


# Set planned_buffer to the boundary-clipped buffer with an explicit area column
planned_buffer = gpd.GeoDataFrame(
    {"OBJECTID":[1], "Name":["BR 421"], "Status":["Proposed"]},
    geometry=intersection,
    crs=planned_road.crs,
)
planned_buffer["area_km2"] = float(buffer_area_km2)

print("columns:", planned_buffer.columns.tolist())
print("area_km2:", float(planned_buffer["area_km2"].iloc[0]))
print("geom type:", planned_buffer.geometry.geom_type.iloc[0])
print("valid:", bool(planned_buffer.geometry.is_valid.all()))
print("planned_buffer total_bounds:", planned_buffer.total_bounds)
print("planned_road total_bounds:", planned_road.total_bounds)
rb, bb = planned_road.total_bounds, planned_buffer.total_bounds
print("buffer extends beyond road on all sides:",
      bb[0]<=rb[0], bb[1]<=rb[1], bb[2]>=rb[2], bb[3]>=rb[3])
print("\nbuffer_area_km2:", float(buffer_area_km2))

try:  # errors in this block are tolerated
    import geopandas as gpd
    import numpy as np

    print("planned_buffer CRS:", planned_buffer.crs, "| rows:", len(planned_buffer), "| types:", planned_buffer.geometry.type.value_counts().to_dict())
    print("deforested_area CRS:", deforested_area.crs, "| rows:", len(deforested_area), "| types:", deforested_area.geometry.type.value_counts().to_dict())
    print("protected_forest CRS:", protected_forest.crs, "| rows:", len(protected_forest))

    # 3a: at-risk area = planned-road buffer (already clipped to state boundary) minus already deforested land
    ar_geom = planned_buffer.geometry.difference(deforested_area.geometry)
    at_risk_area = gpd.GeoDataFrame(geometry=ar_geom, crs=planned_buffer.crs)

    inv = int(at_risk_area.geometry.is_invalid.sum())
    emp = int(at_risk_area.geometry.is_empty.sum())
    print("at_risk geometry types:", at_risk_area.geometry.type.value_counts().to_dict(), "| invalid:", inv, "| empty:", emp)
    if inv:
        at_risk_area.geometry = at_risk_area.geometry.make_valid()
        print("after make_valid, invalid:", int(at_risk_area.geometry.is_invalid.sum()))

    at_risk_area_m2 = float(at_risk_area.geometry.area.sum())
    at_risk_area_km2 = at_risk_area_m2 / 1e6
    print("at_risk_area_km2:", round(at_risk_area_km2, 3))
    print("buffer_area_km2 :", round(buffer_area_km2, 3))
    print("already deforested inside buffer (km2):", round(buffer_area_km2 - at_risk_area_km2, 3))
    print("sanity: 0 < at_risk <= buffer ?", 0 < at_risk_area_km2 <= buffer_area_km2 + 1e-6)
except Exception:
    pass

try:  # errors in this block are tolerated
    import geopandas as gpd
    import numpy as np

    inv = int((~at_risk_area.geometry.is_valid()).sum())
    emp = int(at_risk_area.geometry.is_empty.sum())
    print("at_risk types:", at_risk_area.geometry.type.value_counts().to_dict(), "| invalid:", inv, "| empty:", emp)
    if inv:
        at_risk_area.geometry = at_risk_area.geometry.make_valid()
        print("after make_valid, invalid:", int((~at_risk_area.geometry.is_valid()).sum()))

    at_risk_area_m2 = float(at_risk_area.geometry.area.sum())
    at_risk_area_km2 = at_risk_area_m2 / 1e6
    print("at_risk_area_km2:", round(at_risk_area_km2, 3))
    print("buffer_area_km2 :", round(buffer_area_km2, 3))
    print("already deforested inside buffer (km2):", round(buffer_area_km2 - at_risk_area_km2, 3))
    print("sanity: 0 < at_risk <= buffer ?", 0 < at_risk_area_km2 <= buffer_area_km2 + 1e-6)

    # 3b: protected-forest exposure = overlap of at-risk area with protected forest
    prot_overlap_geom = at_risk_area.geometry.intersection(protected_forest.geometry)
    protected_at_risk = gpd.GeoDataFrame(geometry=prot_overlap_geom, crs=at_risk_area.crs)
    print("protected_at_risk types:", protected_at_risk.geometry.type.value_counts().to_dict(),
          "| empty:", int(protected_at_risk.geometry.is_empty.sum()))

    protected_overlap_m2 = float(protected_at_risk.geometry.area.sum())
    protected_overlap_km2 = protected_overlap_m2 / 1e6
    print("protected_overlap_km2:", round(protected_overlap_km2, 3))
    print("share of at-risk area inside protected forest (%):",
          round(100.0 * protected_overlap_km2 / at_risk_area_km2, 2))

    # how many of the 52 reserves actually intersect
    resv_counts = protected_forest.geometry.intersects(at_risk_area.geometry.iloc[0]).sum()
    print("protected reserves intersecting at-risk area:", int(resv_counts), "of", len(protected_forest))
except Exception:
    pass

try:  # errors in this block are tolerated
    import geopandas as gpd
    import numpy as np

    def empty_mask(gs):
        ie = gs.is_empty
        return ie() if callable(ie) else ie

    inv = int((~at_risk_area.geometry.is_valid()).sum())
    emp = int(empty_mask(at_risk_area.geometry).sum())
    print("at_risk types:", at_risk_area.geometry.type.value_counts().to_dict(), "| invalid:", inv, "| empty:", emp)
    if inv:
        at_risk_area.geometry = at_risk_area.geometry.make_valid()
        print("after make_valid, invalid:", int((~at_risk_area.geometry.is_valid()).sum()))

    at_risk_area_m2 = float(at_risk_area.geometry.area.sum())
    at_risk_area_km2 = at_risk_area_m2 / 1e6
    print("at_risk_area_km2:", round(at_risk_area_km2, 3))
    print("buffer_area_km2 :", round(buffer_area_km2, 3))
    print("already deforested inside buffer (km2):", round(buffer_area_km2 - at_risk_area_km2, 3))
    print("sanity: 0 < at_risk <= buffer ?", 0 < at_risk_area_km2 <= buffer_area_km2 + 1e-6)

    # 3b: protected-forest exposure = overlap of at-risk area with protected forest
    prot_overlap_geom = at_risk_area.geometry.intersection(protected_forest.geometry)
    protected_at_risk = gpd.GeoDataFrame(geometry=prot_overlap_geom, crs=at_risk_area.crs)
    print("protected_at_risk types:", protected_at_risk.geometry.type.value_counts().to_dict(),
          "| empty:", int(empty_mask(protected_at_risk.geometry).sum()))

    protected_overlap_m2 = float(protected_at_risk.geometry.area.sum())
    protected_overlap_km2 = protected_overlap_m2 / 1e6
    print("protected_overlap_km2:", round(protected_overlap_km2, 3))
    print("share of at-risk area inside protected forest (%):",
          round(100.0 * protected_overlap_km2 / at_risk_area_km2, 2))

    resv_counts = int(protected_forest.geometry.intersects(at_risk_area.geometry.iloc[0]).sum())
    print("protected reserves intersecting at-risk area:", resv_counts, "of", len(protected_forest))
except Exception:
    pass

import geopandas as gpd
import numpy as np

inv = int((~at_risk_area.geometry.is_valid).sum())
emp = int(at_risk_area.geometry.is_empty.sum())
print("at_risk types:", at_risk_area.geometry.type.value_counts().to_dict(), "| invalid:", inv, "| empty:", emp)
if inv:
    at_risk_area.geometry = at_risk_area.geometry.make_valid()
    print("after make_valid, invalid:", int((~at_risk_area.geometry.is_valid).sum()))

at_risk_area_m2 = float(at_risk_area.geometry.area.sum())
at_risk_area_km2 = at_risk_area_m2 / 1e6
print("at_risk_area_km2:", round(at_risk_area_km2, 3))
print("buffer_area_km2 :", round(buffer_area_km2, 3))
print("already deforested inside buffer (km2):", round(buffer_area_km2 - at_risk_area_km2, 3))
print("sanity: 0 < at_risk <= buffer ?", 0 < at_risk_area_km2 <= buffer_area_km2 + 1e-6)

# 3b: protected-forest exposure = overlap of at-risk area with protected forest
prot_overlap_geom = at_risk_area.geometry.intersection(protected_forest.geometry)
protected_at_risk = gpd.GeoDataFrame(geometry=prot_overlap_geom, crs=at_risk_area.crs)
print("protected_at_risk types:", protected_at_risk.geometry.type.value_counts().to_dict(),
      "| empty:", int(protected_at_risk.geometry.is_empty.sum()))

protected_overlap_m2 = float(protected_at_risk.geometry.area.sum())
protected_overlap_km2 = protected_overlap_m2 / 1e6
print("protected_overlap_km2:", round(protected_overlap_km2, 3))
print("share of at-risk area inside protected forest (%):",
      round(100.0 * protected_overlap_km2 / at_risk_area_km2, 2))

resv_counts = int(protected_forest.geometry.intersects(at_risk_area.geometry.iloc[0]).sum())
print("protected reserves intersecting at-risk area:", resv_counts, "of", len(protected_forest))

try:  # errors in this block are tolerated
    import geopandas as gpd
    from shapely import get_area

    gs = protected_at_risk.geometry
    print("rows:", len(gs), "| non-null:", int(gs.notna().sum()), "| is_empty sum:", int(gs.is_empty.sum()))
    print("type counts:", gs.type.value_counts(dropna=False).to_dict())
    print("non-null index:", gs.dropna().index.tolist())

    # which of the 52 reserves intersect, and their overlap areas (per-reserve)
    a = at_risk_area.geometry.iloc[0]
    per_res = [get_area(a.intersection(g)) for g in protected_forest.geometry]
    import numpy as np
    per_res = np.array(per_res)
    hit = np.where(per_res > 0)[0]
    print("reserves with overlap area > 0:", len(hit), hit.tolist())
    print("max per-reserve overlap (km2):", per_res.max()/1e6 if len(per_res) else None)
    print("sum of per-reserve overlaps (km2):", per_res.sum()/1e6)

    # robust single-geometry overlap: at_risk vs union of all reserves
    prot_union = protected_forest.geometry.unary_union
    prot_geom = a.intersection(prot_union)
    prot_area_m2 = get_area(prot_geom)
    print("geom type:", prot_geom.geom_type, "| area km2:", prot_area_m2/1e6)

    # also check the 'intersects' flag result vs actual area (touching vs real overlap)
    inter = [bool(a.intersects(g)) for g in protected_forest.geometry]
    print("reserves with intersects=True:", int(sum(inter)))
except Exception:
    pass

import geopandas as gpd
import numpy as np
from shapely import area as shape_area

gs = protected_at_risk.geometry
print("rows:", len(gs), "| non-null:", int(gs.notna().sum()), "| is_empty sum:", int(gs.is_empty.sum()))
print("type counts:", gs.type.value_counts(dropna=False).to_dict())
print("non-null index:", gs.dropna().index.tolist())

a = at_risk_area.geometry.iloc[0]
per_res = np.array([shape_area(a.intersection(g)) for g in protected_forest.geometry])
hit = np.where(per_res > 0)[0]
print("reserves with overlap area > 0:", len(hit), hit.tolist())
print("max per-reserve overlap (km2):", per_res.max()/1e6)
print("sum of per-reserve overlaps (km2):", per_res.sum()/1e6)

# robust single-geometry overlap: at_risk vs union of all reserves
prot_union = protected_forest.geometry.unary_union
prot_geom = a.intersection(prot_union)
prot_area_m2 = float(shape_area(prot_geom))
print("union-intersection geom type:", prot_geom.geom_type, "| area km2:", prot_area_m2/1e6)

inter = [bool(a.intersects(g)) for g in protected_forest.geometry]
print("reserves with intersects=True:", int(sum(inter)))

import geopandas as gpd
from shapely import area as shape_area
from shapely.validation import make_valid

# clean protected_at_risk: single geometry, no empty parts
prot_geom = at_risk_area.geometry.iloc[0].intersection(protected_forest.geometry.unary_union)
prot_geom = make_valid(prot_geom)
protected_at_risk = gpd.GeoDataFrame(geometry=[prot_geom], crs=at_risk_area.crs)

protected_overlap_km2 = float(shape_area(prot_geom)) / 1e6
at_risk_area_km2 = float(at_risk_area.geometry.area.sum()) / 1e6

print("=== STEP 3 RESULTS (equal-area CRS ESRI:102033, metres) ===")
print("planned road length          : %.1f km" % (planned_length_m/1e3))
print("buffer area (inside boundary): %.1f km2" % buffer_area_km2)
print("at-risk area (new deforestation) = buffer - deforested : %.1f km2" % at_risk_area_km2)
print("  already deforested inside buffer (excluded)          : %.1f km2" % (buffer_area_km2 - at_risk_area_km2))
print("protected forest exposed within at-risk area           : %.1f km2" % protected_overlap_km2)
print("  share of at-risk area in protected forest            : %.1f%%" % (100*protected_overlap_km2/at_risk_area_km2))
print("protected reserves intersecting at-risk area           : 2 of 52")
print()
print("at_risk_area        : %s, %d row(s), valid=%s, empty=%s" % (
    at_risk_area.geometry.type.tolist(), len(at_risk_area),
    bool(at_risk_area.geometry.is_valid.all()), bool(at_risk_area.geometry.is_empty.all())))
print("protected_at_risk   : %s, %d row(s), valid=%s" % (
    protected_at_risk.geometry.type.tolist(), len(protected_at_risk),
    bool(protected_at_risk.geometry.is_valid.all())))
print("CRS at_risk / protected_at_risk:", at_risk_area.crs.to_epsg(), "/", protected_at_risk.crs.to_epsg())
print("sanity 0 < at_risk <= buffer:", 0 < at_risk_area_km2 <= buffer_area_km2 + 1e-6,
      "| 0 < prot_overlap <= at_risk:", 0 <= protected_overlap_km2 <= at_risk_area_km2 + 1e-6)

print("at_risk_area CRS:", at_risk_area.crs)
print("protected_at_risk CRS:", protected_at_risk.crs)
names = protected_forest.loc[[14, 15], ["Name", "Category"]].values.tolist()
print("exposed reserves:", names)

import time
t1 = time.time()

# 1) Confirm all layers share the same projected CRS
layers = {
    'state_boundary': state_boundary,
    'protected_forest': protected_forest,
    'deforested_area': deforested_area,
    'planned_buffer': planned_buffer,
    'at_risk_area': at_risk_area,
    'planned_road': planned_road,
    'roads': roads,
}
crs_map = {k: (str(v.crs.to_epsg()) if v.crs.to_epsg() else str(v.crs)) for k, v in layers.items()}
for k, v in crs_map.items():
    print(f'{k}: {v}')
assert len(set(crs_map.values())) == 1, 'CRS mismatch between layers!'
print('All layers share the same projected CRS:', crs_map['roads'])
print('target_crs:', target_crs)

# 2) Simplify existing roads for fast, clean plotting (tolerance in metres)
tol_m = 10.0
roads_plot = roads.copy()
roads_plot['geometry'] = roads_plot['geometry'].simplify(tol_m, preserve_topology=True)
n_before = len(roads_plot)
roads_plot = roads_plot[~roads_plot['geometry'].is_empty & roads_plot['geometry'].is_valid].reset_index(drop=True)
roads_plot = roads_plot.set_crs(target_crs)
n_after = len(roads_plot)
print(f'road features before simplify: {n_before}, after: {n_after} (dropped {n_before - n_after} empty/invalid)')

def _npts(gs):
    tot = 0
    for g in gs:
        if g.geom_type == 'LineString':
            tot += len(g.coords)
        else:
            tot += sum(len(p.coords) for p in g.geoms)
    return tot
n_roads_pts_after = _npts(roads_plot.geometry)
print(f'road coordinates: {n_roads_pts} -> {n_roads_pts_after} (simplify tolerance {tol_m} m)')
print('roads_plot ready:', len(roads_plot), 'geometries, CRS =', roads_plot.crs)
print('elapsed so far: %.1f s' % (time.time() - t1))

# 3) Define plot extent: bounding box of state boundary + a ~3% padding margin
import numpy as np
state_bounds = state_boundary.total_bounds  # minx, miny, maxx, maxy
minx, miny, maxx, maxy = state_bounds
w, h = maxx - minx, maxy - miny
pad_x, pad_y = 0.03 * w, 0.03 * h
plot_extent = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
print('state boundary bbox (m):', state_bounds)
print('plot extent with 3% padding:', plot_extent)
print('plot size: %.0f km x %.0f km' % ((plot_extent[2]-plot_extent[0])/1000, (plot_extent[3]-plot_extent[1])/1000))

# 4) Define visual style order (bottom -> top) and styles
plot_order = [
    ('deforested_area',  {'color': '#5b2c0c', 'label': 'Already deforested area', 'alpha': 0.85}),
    ('protected_forest', {'color': '#2e7d32', 'label': 'Protected forest (reservations)', 'alpha': 0.55}),
    ('at_risk_area',     {'color': '#d62728', 'label': 'At-risk area (planned-road buffer not yet deforested)', 'alpha': 0.5}),
    ('planned_buffer',   {'color': '#ff7f0e', 'label': 'Planned road buffer (5.5 km)', 'alpha': 0.0, 'edgecolor': '#ff7f0e', 'linestyle': '--', 'linewidth': 1.5}),
    ('roads_plot',       {'color': '#888888', 'label': 'Existing roads', 'linewidth': 0.5, 'alpha': 0.6}),
    ('planned_road',     {'color': '#e31536', 'label': 'Planned road (BR 421)', 'linewidth': 2.5}),
    ('state_boundary',   {'color': 'black', 'label': 'Rondônia boundary (approx.)', 'linewidth': 2.0}),
]
print('\nVisual style order (bottom -> top):')
for i, (name, st) in enumerate(plot_order, 1):
    print(f'  {i}. {name}: {st}')

# 5) Final readiness check: all plot inputs exist, valid CRS, non-empty geometry
for name, _ in plot_order:
    gdf = globals()[name]
    n_geom = len(gdf)
    n_empty = int(gdf.geometry.is_empty.sum())
    print(f'{name}: {n_geom} features, {n_empty} empty, CRS ok = {gdf.crs == target_crs}')

print('\nRoad geometries prepared for plotting:', len(roads_plot))
print('Step 4 complete.')

print('=== Step 4 deliverables in environment ===')
print('roads_plot        :', len(roads_plot), 'road geometries (simplified 236,835 -> 206,979 coords @10 m), CRS =', roads_plot.crs.to_authority()[0] if roads_plot.crs.to_authority() else roads_plot.crs)
print('planned_buffer    :', len(planned_buffer), 'geom, area = %.1f km2' % buffer_area_km2)
print('at_risk_area      :', len(at_risk_area), 'geom, area = %.1f km2' % at_risk_area_km2)
print('protected_forest  :', len(protected_forest), 'geoms')
print('deforested_area   :', len(deforested_area), 'geom')
print('planned_road      :', len(planned_road), 'geom')
print('state_boundary    :', len(state_boundary), 'geom')
print('plot_extent       :', tuple(round(x, 1) for x in plot_extent))
print('plot_order (z-bottom->top):', [n for n, _ in plot_order])
print('All layers CRS-consistent: True; roads ready for plotting: 27662 geometries.')

import os, shapely, geopandas as gpd

# --- check step-4 variables ---
try:
    _ = roads_plot
    print('roads_plot exists, rows:', len(roads_plot), 'crs:', roads_plot.crs)
except NameError:
    roads_plot = roads
    print('roads_plot NOT defined -> created from roads, rows:', len(roads_plot))

try:
    print('plot_extent exists:', plot_extent)
except NameError:
    print('plot_extent NOT defined -> will compute from state_boundary')

# --- merge all existing road segments into one MultiLineString for fast plotting ---
parts = []
for g in roads_plot.geometry:
    if g is None or g.is_empty:
        continue
    if g.geom_type == 'LineString':
        parts.append(tuple(g.coords))
    else:  # MultiLineString
        for sub in g.geoms:
            parts.append(tuple(sub.coords))
roads_line = gpd.GeoDataFrame(geometry=[shapely.geometry.MultiLineString(parts)], crs=roads_plot.crs)
print('merged line parts:', len(parts))

# --- extents for layout decisions ---
print('state bounds   (xmin,ymin,xmax,ymax):', state_boundary.total_bounds)
print('planned bounds:', planned_road.total_bounds)
print('at_risk bounds:', at_risk_area.total_bounds)
print('deforested     :', deforested_area.total_bounds)
print('protected      :', protected_forest.total_bounds)
print('areas km2: buffer', round(buffer_area_km2,1), '| at-risk', round(at_risk_area_km2,1), '| prot overlap', round(protected_overlap_km2,1))

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from shapely.geometry import box as shbox

os.makedirs('pred_results', exist_ok=True)

# ---------- helpers ----------
def seg_array(geometry_iter):
    """Concatenate line segments into one (N,2) array with NaN gaps between segments."""
    rows = []
    nseg = 0
    for g in geometry_iter:
        if g is None or g.is_empty:
            continue
        lines = g.geoms if g.geom_type == 'MultiLineString' else [g]
        for ln in lines:
            c = np.asarray(ln.coords)
            for i in range(len(c) - 1):
                rows.append(c[i]); rows.append(c[i+1])
                nseg += 1
                if nseg % 1 == 0:
                    rows.append(np.full(2, np.nan))
    arr = np.array(rows[:-1])
    return arr

roads_arr = seg_array(roads_plot.geometry)          # ~237k segments -> one path
print('roads_arr shape:', roads_arr.shape)

bnd = state_boundary.geometry.iloc[0]
bnd_xy = np.asarray(bnd.exterior.coords)

planned_xy = np.asarray(planned_road.geometry.iloc[0].coords)
print('planned road vertices:', len(planned_xy))

# inset extent: planned road bbox padded 150 km
px0, py0, px1, py1 = planned_road.total_bounds
pad = 150000.0
inset = (px0 - pad, py0 - pad, px1 + pad, py1 + pad)
inset_box = shbox(*inset)
in_idx = list(roads_plot.sindex.query(inset_box))
roads_in_arr = seg_array(roads_plot.geometry.iloc[in_idx])
print('inset road features:', len(in_idx), 'inset road arr shape:', roads_in_arr.shape)

def km(v): return f'{v/1000:,.0f}'

# ---------- figure ----------
fig = plt.figure(figsize=(15, 11), facecolor='white')
ax1 = fig.add_axes([0.04, 0.14, 0.86, 0.78])
ax2 = fig.add_axes([0.575, 0.17, 0.30, 0.212])   # inset, aspect-matched to extent

ex = plot_extent
m = 0.005
ex = (ex[0]-m*(ex[2]-ex[0]), ex[1]-m*(ex[3]-ex[1]), ex[2]+m*(ex[2]-ex[0]), ex[3]+m*(ex[3]-ex[1]))

def draw_map(ax):
    deforested_area.plot(ax=ax, facecolor='#d9d9d9', edgecolor='#9e9e9e', linewidth=0.2)
    protected_forest.plot(ax=ax, facecolor='#a8d5a2', edgecolor='#38761d', linewidth=0.4, alpha=0.95)
    planned_buffer.plot(ax=ax, facecolor='#b3d4f0', alpha=0.45, edgecolor='#1f77b4',
                        linewidth=0.8, linestyle='--')
    at_risk_area.plot(ax=ax, facecolor='#e45756', alpha=0.75, edgecolor='#8b0000', linewidth=0.6)
    ax.plot(roads_arr[:,0], roads_arr[:,1], color='#555555', linewidth=0.18, zorder=5)
    ax.plot(planned_xy[:,0], planned_xy[:,1], color='#1f77b4', linewidth=2.5, zorder=6, solid_capstyle='round')
    ax.plot(bnd_xy[:,0], bnd_xy[:,1], color='#333333', linewidth=1.6, zorder=7)

draw_map(ax1)
ax1.set_xlim(ex[0], ex[2]); ax1.set_ylim(ex[1], ex[3])
ax1.set_aspect('equal')
ax1.set_title('Predicted new-deforestation risk from the planned road (BR 421, "Proposed") — Rondônia\n'
              '5.5 km buffer = 2,343.1 km²   |   at-risk (not yet deforested) = 1,349.7 km², '
              'of which 790.6 km² lies inside protected forest',
              fontsize=10.5, loc='center', pad=8)
ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, p: km(v)))
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: km(v)))
ax1.set_xlabel('Easting (km)'); ax1.set_ylabel('Northing (km)')

# inset
draw_map(ax2)
ax2.plot(roads_in_arr[:,0], roads_in_arr[:,1], color='#555555', linewidth=0.5, zorder=5)
ax2.set_xlim(inset[0], inset[2]); ax2.set_ylim(inset[1], inset[3])
ax2.set_aspect('equal')
ax2.set_title('Planned road & 5.5 km buffer (zoom)', fontsize=9, pad=4)
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, p: km(v)))
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: km(v)))
for s in ax2.spines.values(): s.set_edgecolor('#333333')

# ---------- legend ----------
handles = [
    Patch(fc='#d9d9d9', ec='#9e9e9e', label='Deforested area'),
    Patch(fc='#a8d5a2', ec='#38761d', label='Protected forest'),
    Patch(fc='#b3d4f0', ec='#1f77b4', ls='--', alpha=0.7, label='Planned road 5.5 km buffer'),
    Patch(fc='#e45756', ec='#8b0000', alpha=0.85, label='At-risk area (new deforestation)'),
    Line2D([0], [0], color='#555555', lw=1.0, label='Existing roads'),
    Line2D([0], [0], color='#1f77b4', lw=2.5, label='Planned road (BR 421)'),
    Line2D([0], [0], color='#333333', lw=1.6, label='Rondônia boundary (approx.)'),
]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8.5,
           bbox_to_anchor=(0.5, 0.025), frameon=True)

out_path = 'pred_results/predictedRiskyArea.png'
fig.savefig(out_path, dpi=200)
plt.close(fig)

sz = os.path.getsize(out_path)
print('saved:', os.path.abspath(out_path), '| size:', sz, 'bytes | exists:', os.path.exists(out_path))

import numpy as np
from PIL import Image
import os

im = Image.open('pred_results/predictedRiskyArea.png').convert('RGB')
a = np.asarray(im)
print('image size (WxH):', a.shape)
px = a.reshape(-1, 3).astype(int)

def near(c, tol=40):
    d = np.abs(px - np.array(c)).sum(axis=1)
    return int((d < tol).sum())

checks = {
    'white bg (~255)':        near((255, 255, 255)),
    'deforested grey (~217)': near((217, 217, 217)),
    'protected green (~168,213,162)': near((168, 213, 162)),
    'buffer blue (~179,212,240)':     near((179, 212, 240)),
    'at-risk red (~228,87,86)':       near((228, 87, 86)),
    'planned road blue (~31,119,180)':near((31, 119, 180)),
    'road grey (~85)':               near((85, 85, 85)),
    'boundary dark grey (~51)':      near((51, 51, 51)),
}
for k, v in checks.items():
    print(f'{k:38s}: {v:8d} px')

# uniform-image guard: count distinct colors
u, c = np.unique(px, axis=0, return_counts=True)
print('distinct colors:', len(u), '| unique-pixel fraction:', round(1 - c.max()/px.size, 4))
print('file exists:', os.path.exists('pred_results/predictedRiskyArea.png'),
      '| size bytes:', os.path.getsize('pred_results/predictedRiskyArea.png'))

import numpy as np
from PIL import Image
a = np.asarray(Image.open('pred_results/predictedRiskyArea.png').convert('RGB')).reshape(-1,3).astype(int)

def near(c, tol=30):
    return int((np.abs(a - np.array(c)).sum(axis=1) < tol).sum())

# red alpha=0.75 blended over deforested grey(217), protected green(168,213,162), buffer-blue, white
print('red over grey   ~(225,120,120):', near((225,120,120)))
print('red over green  ~(199,119,119):', near((199,119,119), 25))
print('red over blue   ~(204,150,178):', near((204,150,178), 25))
# buffer blue alpha=0.45 blended
print('buffer over white ~(221,236,248):', near((221,236,248)))
print('buffer over grey  ~(218,233,246):', near((218,233,246), 25))
print('buffer over green ~(173,213,197):', near((173,213,197), 25))

