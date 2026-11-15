# Correct: the layers are built in a loop into a dict and then tuple-unpacked before plotting.
import geopandas as gpd
from shapely.validation import make_valid
roads_raw = gpd.read_file("dataset/roads.geojson")
defo_raw = gpd.read_file("dataset/deforestedArea.geojson")
prot_raw = gpd.read_file("dataset/protectedForest.geojson")
planned_raw = gpd.read_file("dataset/planned_road.geojson")
target = planned_raw.crs
def fix(g):
    g = g.copy()
    n0 = (~g.geometry.is_valid).sum()
    g["geometry"] = g.geometry.apply(lambda x: make_valid(x) if x is not None and not x.is_valid else x)
    return g, n0
out = {}
for name, g in [("roads", roads_raw), ("deforested", defo_raw), ("protected", prot_raw), ("planned", planned_raw)]:
    g2, n0 = fix(g)
    g2 = g2.to_crs(target)
    g2, n1 = fix(g2)
    out[name] = g2
    print(name, "| invalid before:", n0, "| after reproj fix:", (~g2.geometry.is_valid).sum(), "| n:", len(g2),
          "| geom types:", g2.geom_type.value_counts().to_dict(), "| null:", g2.geometry.isna().sum())
roads, deforested, protected, planned = out["roads"], out["deforested"], out["protected"], out["planned"]
print(target.name, roads.crs == deforested.crs == protected.crs == planned.crs)
print("deforested area km2:", deforested.area.sum()/1e6, "protected km2:", protected.area.sum()/1e6, "planned length km:", planned.length.sum()/1e3)

from shapely.geometry import MultiPolygon, Polygon
def polys_only(geom):
    if isinstance(geom,(Polygon,MultiPolygon)): return geom
    parts=[]
    for p in getattr(geom,"geoms",[]):
        if isinstance(p,Polygon): parts.append(p)
        elif isinstance(p,MultiPolygon): parts.extend(p.geoms)
    return MultiPolygon(parts)
print([type(p).__name__ for p in deforested.geometry.iloc[0].geoms][:10], len(deforested.geometry.iloc[0].geoms))
deforested["geometry"]=deforested.geometry.apply(polys_only)
print(deforested.geom_type.tolist(), deforested.is_valid.all(), deforested.area.sum()/1e6)

import geopandas as gpd
print(planned.crs, planned.length.values/1000)
buf = planned.buffer(5500)
planned_buffer = gpd.GeoDataFrame({'Name':['planned_buffer_5.5km']}, geometry=[buf.union_all()], crs=planned.crs)
print(len(planned_buffer), planned_buffer.geom_type.values, planned_buffer.is_valid.values)
print("Buffer area km2:", planned_buffer.area.values[0]/1e6)


import geopandas as gpd
bgeom = planned_buffer.geometry.union_all() if hasattr(planned_buffer.geometry,'union_all') else planned_buffer.unary_union
defo_clip = gpd.clip(deforested, planned_buffer)
prot_clip = gpd.clip(protected, planned_buffer)
defo_clip = defo_clip[~defo_clip.is_empty]
prot_clip = prot_clip[~prot_clip.is_empty]
print("deforested clip rows:", len(defo_clip), "protected clip rows:", len(prot_clip), prot_clip['Name'].tolist() if len(prot_clip) else [])

step1 = gpd.overlay(planned_buffer[['geometry']], defo_clip[['geometry']], how='difference')
risky_area = gpd.overlay(step1, prot_clip[['geometry']], how='difference') if len(prot_clip) else step1
risky_area = risky_area[~risky_area.is_empty]

buf_km2 = planned_buffer.area.sum()/1e6
defo_km2 = defo_clip.dissolve().area.sum()/1e6 if len(defo_clip) else 0
prot_km2 = prot_clip.dissolve().area.sum()/1e6 if len(prot_clip) else 0
overlap_km2 = (defo_clip.union_all().intersection(prot_clip.union_all()).area/1e6) if len(defo_clip) and len(prot_clip) else 0
risk_km2 = risky_area.area.sum()/1e6
print(f"Buffer: {buf_km2:.1f} km2")
print(f"Already deforested in buffer: {defo_km2:.1f} km2")
print(f"Protected in buffer: {prot_km2:.1f} km2 (overlap defo&prot {overlap_km2:.1f})")
print(f"At-risk area: {risk_km2:.1f} km2; check: {buf_km2-defo_km2-prot_km2+overlap_km2:.1f}")
print("risky rows:", len(risky_area), risky_area.geom_type.value_counts().to_dict(), "valid:", risky_area.is_valid.all())



import shapely
from shapely.ops import unary_union
import geopandas as gpd
parts = [deforested.geometry.union_all(), protected.geometry.union_all(),
         roads.geometry.buffer(100).union_all()]
allgeom = unary_union(parts)
hull = allgeom.convex_hull
print("convex hull km2:", hull.area/1e6)
for r in [0.05, 0.1, 0.2, 0.3]:
    ch = shapely.concave_hull(allgeom, ratio=r)
    print(r, ch.geom_type, ch.area/1e6)



bnd_geom = shapely.concave_hull(allgeom, ratio=0.2).buffer(1000)
boundary = gpd.GeoDataFrame({"Name":["Rondonia (approx.)"]}, geometry=[bnd_geom], crs=roads.crs)
print("boundary valid:", boundary.is_valid.all(), "type:", bnd_geom.geom_type)
print("boundary area km2: %.0f (reference ~237,600; diff %.1f%%)" % (bnd_geom.area/1e6, (bnd_geom.area/1e6-237600)/237600*100))
roads_clipped = gpd.clip(roads, boundary)
print("roads before/after clip:", len(roads), len(roads_clipped))
print("road length km before/after: %.0f / %.0f" % (roads.length.sum()/1e3, roads_clipped.length.sum()/1e3))
print("planned road within boundary:", planned.geometry.iloc[0].within(bnd_geom), "| buffer within:", planned_buffer.geometry.iloc[0].within(bnd_geom))



bnd_geom = allgeom.convex_hull.buffer(1000)   # fixed 1 km outer buffer, not tuned
boundary = gpd.GeoDataFrame({"Name":["Rondonia (approx., convex hull)"]}, geometry=[bnd_geom], crs=roads.crs)
a = bnd_geom.area/1e6
print("boundary valid:", boundary.is_valid.all(), "type:", bnd_geom.geom_type)
print("convex hull (no buffer) km2: %.0f" % (allgeom.convex_hull.area/1e6))
print("boundary area km2: %.0f vs reference ~237,600 -> diff %+.0f km2 (%+.1f%%)" % (a, a-237600, (a-237600)/237600*100))
print("data footprint union area km2: %.0f" % (allgeom.area/1e6))
roads_clipped = gpd.clip(roads, boundary)
print("roads before/after clip:", len(roads), len(roads_clipped))
print("road length km before/after: %.0f / %.0f" % (roads.length.sum()/1e3, roads_clipped.length.sum()/1e3))
print("planned road within:", planned.geometry.iloc[0].within(bnd_geom), "| buffer within:", planned_buffer.geometry.iloc[0].within(bnd_geom))



import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
os.makedirs("pred_results", exist_ok=True)
print("risky empty:", risky_area.is_empty.all(), "risk km2:", round(risk_km2,1))

fig, axes = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={"width_ratios":[1.15,1]})
def draw(ax, lw_road):
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.5, zorder=1)
    protected.plot(ax=ax, color="#2e8b57", alpha=0.6, edgecolor="#1b5e20", linewidth=0.3, zorder=2)
    deforested.plot(ax=ax, color="#d2b48c", edgecolor="none", zorder=3)
    roads_clipped.plot(ax=ax, color="grey", linewidth=lw_road, zorder=4)
    risky_area.plot(ax=ax, color="#ff00ff", alpha=0.85, edgecolor="none", zorder=5)
    planned_buffer.boundary.plot(ax=ax, color="black", linestyle="--", linewidth=1.2, zorder=6)
    planned.plot(ax=ax, color="red", linewidth=2, zorder=7)
    ax.set_xlabel("Easting (m, South America Albers Equal Area)")
    ax.set_ylabel("Northing (m)")
draw(axes[0], 0.2)
axes[0].set_title("Rondônia – planned road BR 421 and predicted deforestation risk")
draw(axes[1], 0.5)
minx, miny, maxx, maxy = planned_buffer.total_bounds
pad = 25000
axes[1].set_xlim(minx-pad, maxx+pad); axes[1].set_ylim(miny-pad, maxy+pad)
axes[1].set_title(f"Zoom: 5.5 km buffer – at-risk area {risk_km2:,.0f} km² of {buf_km2:,.0f} km²")
# zoom box on overview
from shapely.geometry import box
gpd_box = planned_buffer.copy(); gpd_box["geometry"] = [box(minx-pad, miny-pad, maxx+pad, maxy+pad)]
gpd_box.boundary.plot(ax=axes[0], color="blue", linewidth=1.2, zorder=8)
handles = [Line2D([0],[0],color="black",lw=1.5,label="Rondônia boundary (approx.)"),
           Patch(facecolor="#2e8b57",alpha=0.6,label="Protected forest"),
           Patch(facecolor="#d2b48c",label="Deforested area"),
           Line2D([0],[0],color="grey",lw=1,label="Existing roads"),
           Line2D([0],[0],color="red",lw=2,label="Planned road (BR 421)"),
           Line2D([0],[0],color="black",lw=1.2,ls="--",label="5.5 km buffer"),
           Patch(facecolor="#ff00ff",alpha=0.85,label="At risk of new deforestation"),
           Line2D([0],[0],color="blue",lw=1.2,label="Zoom extent")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11, frameon=True)
fig.suptitle("Predicted effect of the proposed road on deforestation in Rondônia", fontsize=16)
plt.tight_layout(rect=[0,0.08,1,0.95])
out_path = "pred_results/predictedRiskyArea.png"
plt.savefig(out_path, dpi=150); plt.close()
print(out_path, os.path.exists(out_path), os.path.getsize(out_path))
