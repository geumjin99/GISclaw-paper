# Correct: the context layers reach the plot through a conditional expression and a helper function; map
# with an inset zoom.

import geopandas as gpd
from shapely.validation import make_valid
roads_raw = gpd.read_file("dataset/roads.geojson")
defo_raw = gpd.read_file("dataset/deforestedArea.geojson")
prot_raw = gpd.read_file("dataset/protectedForest.geojson")
planned = gpd.read_file("dataset/planned_road.geojson")
target = planned.crs
print(target)
def fix(g):
    g = g.copy()
    print("invalid before:", (~g.is_valid).sum())
    g["geometry"] = g.geometry.apply(lambda x: make_valid(x) if not x.is_valid else x)
    return g
deforested = fix(defo_raw).to_crs(target)
protected = fix(prot_raw).to_crs(target)
roads = roads_raw.to_crs(target)
# re-check after reprojection; fix again if needed
for n, g in [("deforested", deforested), ("protected", protected)]:
    inv = (~g.is_valid).sum()
    if inv:
        g["geometry"] = g.geometry.apply(lambda x: make_valid(x) if not x.is_valid else x)
for n, g in [("roads", roads), ("deforested", deforested), ("protected", protected), ("planned", planned)]:
    print(n, len(g), g.crs.name, "invalid:", (~g.is_valid).sum(), "types:", g.geom_type.value_counts().to_dict(), "bounds:", g.total_bounds.round(0))



from shapely.geometry import MultiPolygon, Polygon
gc = deforested.geometry.iloc[0]
print({t: sum(1 for p in gc.geoms if p.geom_type==t) for t in set(p.geom_type for p in gc.geoms)})
polys = []
for p in gc.geoms:
    if p.geom_type == "Polygon": polys.append(p)
    elif p.geom_type == "MultiPolygon": polys.extend(p.geoms)
mp = MultiPolygon(polys)
print("area km2 before/after:", gc.area/1e6, mp.area/1e6, "valid:", mp.is_valid)
deforested = deforested.set_geometry([mp], crs=target)
print(deforested.geom_type.tolist(), (~deforested.is_valid).sum(), "protected area km2:", protected.area.sum()/1e6)


print(planned.crs, planned.geometry.length.values)
planned_buffer = planned[['geometry']].copy()
planned_buffer['geometry'] = planned.geometry.buffer(5500)
planned_buffer = planned_buffer.dissolve().reset_index(drop=True)
print(len(planned_buffer), planned_buffer.geom_type.values, planned_buffer.is_valid.values)
print("Buffer area km2:", planned_buffer.area.sum()/1e6)
print("Expected approx:", (2*5.5*planned.length.sum()/1000 + 3.14159*5.5**2))
print(planned_buffer.total_bounds)

import geopandas as gpd
defo_in = gpd.clip(deforested, planned_buffer)
prot_in = gpd.clip(protected, planned_buffer)
step1 = gpd.overlay(planned_buffer, deforested[['geometry']], how='difference')
risky_area = gpd.overlay(step1, protected[['geometry']], how='difference')
risky_area = risky_area[~risky_area.is_empty]
buf_a = planned_buffer.area.sum()/1e6
d_a = defo_in.area.sum()/1e6
p_a = prot_in.union_all().area/1e6
r_a = risky_area.area.sum()/1e6
overlap = defo_in.union_all().intersection(prot_in.union_all()).area/1e6
print(f"buffer {buf_a:.1f} km2, deforested in buffer {d_a:.1f}, protected in buffer {p_a:.1f}, overlap {overlap:.1f}, at-risk {r_a:.1f}")
print(f"check: {buf_a - d_a - p_a + overlap:.1f}")
print(f"at-risk share {100*r_a/buf_a:.1f}%")
print(len(risky_area), risky_area.geom_type.value_counts().to_dict(), risky_area.is_valid.all())


import shapely
from shapely.ops import unary_union
allg = unary_union([deforested.geometry.unary_union.convex_hull,
                    protected.geometry.unary_union.convex_hull,
                    roads.geometry.unary_union.convex_hull])
hull = allg.convex_hull
print("convex hull area km2:", hull.area/1e6)
# concave hull alternative from all vertices
pts = shapely.MultiPoint(shapely.get_coordinates(shapely.GeometryCollection([deforested.geometry.iloc[0], roads.geometry.unary_union, protected.geometry.unary_union])))
print("n pts:", len(pts.geoms))
for r in [0.05, 0.1, 0.2]:
    ch = shapely.concave_hull(pts, ratio=r)
    print(r, ch.geom_type, ch.area/1e6, ch.is_valid)



import geopandas as gpd
bnd_geom = shapely.concave_hull(pts, ratio=0.2)
boundary = gpd.GeoDataFrame({"name":["Rondonia (approx.)"]}, geometry=[bnd_geom], crs=roads.crs)
print("boundary area km2:", boundary.area.iloc[0]/1e6, "valid:", bnd_geom.is_valid)
print("bounds:", boundary.total_bounds)
print("buffer inside boundary share:", planned_buffer.geometry.iloc[0].intersection(bnd_geom).area/planned_buffer.area.iloc[0])
print("planned road inside share:", planned.geometry.iloc[0].intersection(bnd_geom).length/planned.length.iloc[0])
roads_clip = gpd.clip(roads, boundary)
print("roads before/after clip:", len(roads), len(roads_clip), "length km:", roads.length.sum()/1e3, roads_clip.length.sum()/1e3)



import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
os.makedirs("pred_results", exist_ok=True)

def draw(ax, lw_roads=0.2):
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=1)
    deforested.plot(ax=ax, color="tan", alpha=0.8, zorder=2)
    protected.plot(ax=ax, color="forestgreen", alpha=0.6, zorder=3)
    roads_clip.plot(ax=ax, color="grey", linewidth=lw_roads, zorder=4)
    risky_area.plot(ax=ax, color="orangered", alpha=0.85, zorder=5)
    planned_buffer.boundary.plot(ax=ax, color="black", linestyle="--", linewidth=1, zorder=6)
    planned.plot(ax=ax, color="red", linewidth=2.5, zorder=7)

fig, ax = plt.subplots(figsize=(13, 11))
draw(ax)
ax.set_title("Predicted area at risk of new deforestation\nwithin 5.5 km of planned road BR 421, Rondônia", fontsize=14)
ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
ax.text(0.01, 0.01, "CRS: South America Albers Equal Area Conic (ESRI:102033)\n"
        f"Buffer {buf_a:,.0f} km² | deforested {d_a:,.0f} km² | protected {p_a:,.0f} km² | at risk {r_a:,.0f} km² ({100*r_a/buf_a:.1f}%)",
        transform=ax.transAxes, fontsize=9, va="bottom", bbox=dict(fc="white", alpha=0.8))
handles = [Line2D([], [], color="black", lw=1.2, label="Rondônia boundary (approx.)"),
           Patch(fc="tan", label="Deforested area"),
           Patch(fc="forestgreen", alpha=0.6, label="Protected forest"),
           Line2D([], [], color="grey", lw=1, label="Existing roads"),
           Line2D([], [], color="red", lw=2.5, label="Planned road (BR 421)"),
           Line2D([], [], color="black", ls="--", lw=1, label="5.5 km buffer"),
           Patch(fc="orangered", alpha=0.85, label="At-risk area")]
ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9)

# inset zoom
axin = ax.inset_axes([0.0, 0.62, 0.38, 0.30])
draw(axin, lw_roads=0.4)
minx, miny, maxx, maxy = planned_buffer.total_bounds
pad = 10000
axin.set_xlim(minx-pad, maxx+pad); axin.set_ylim(miny-pad, maxy+pad)
axin.set_xticks([]); axin.set_yticks([]); axin.set_title("Zoom: buffer region", fontsize=9)
ax.indicate_inset_zoom(axin, edgecolor="black")
out = "pred_results/predictedRiskyArea.png"
plt.savefig(out, dpi=200, bbox_inches="tight"); plt.close()
print(os.path.exists(out), os.path.getsize(out))



bx0,by0,bx1,by1 = boundary.total_bounds
print(boundary.total_bounds)
print("buffer x frac", (minx-bx0)/(bx1-bx0), (maxx-bx0)/(bx1-bx0), "y frac", (miny-by0)/(by1-by0), (maxy-by0)/(by1-by0))
