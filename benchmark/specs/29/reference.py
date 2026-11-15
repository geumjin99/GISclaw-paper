"""Reference solution for task 29: tsunami travel time to Virginia Beach over Atlantic bathymetry.

Open-source version of the Learn ArcGIS lesson 'Calculate travel time for a tsunami':
speed = Int(sqrt(g * depth)) from the inverted DEM, geodesically densified source->Virginia
Beach paths, speed sampled along each path, segment time = length / speed, summed per route.
In addition (instruction: 'simulate wave travel over the bathymetric surface to create an
arrival time map') an arrival-time surface is computed for each source as the least-time path
over the speed grid (Huygens / fastest-path approximation, skimage MCP)."""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import CRS, Geod, Transformer
from skimage.graph import MCP_Geometric
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.80665
geod = Geod(ellps="WGS84")

# ---------------------------------------------------------------- 1. data
with rasterio.open("dataset/AtlanticDEM.tif") as src:
    dem = src.read(1, masked=True).filled(np.nan).astype("float64")
    profile = src.profile.copy()
    transform, raster_crs = src.transform, src.crs
crs = CRS.from_wkt(raster_crs.to_wkt())                 # Atlantic_Albers_EqualArea_Custom, metres
cell = transform.a                                       # 5000 m
# The GeoJSON files carry the coordinates of the project CRS (metres) but no 'crs' member, so
# readers default to EPSG:4326. Assign the DEM's CRS (do NOT reproject).
paths = gpd.read_file("dataset/TsuniamiPaths.gpkg").set_crs(crs, allow_override=True)
sources = gpd.read_file("dataset/TsunamiSources.gpkg").set_crs(crs, allow_override=True)
countries = gpd.read_file("dataset/World_Countries.gpkg")
print("routes:", paths["Route"].tolist())

# ---------------------------------------------------------------- 2-3. depth -> speed (m/s)
depth = -dem                                             # invert: ocean depths positive
speed = np.full(dem.shape, np.nan)
ocean = np.isfinite(depth) & (depth > 0)
speed[ocean] = np.floor(np.sqrt(G * depth[ocean]))       # Int(SquareRoot(9.80665 * -DEM))
speed[speed <= 0] = np.nan                               # depth < 0.1 m -> speed 0: no travel
print("speed m/s: min %.0f max %.0f median %.0f" % (np.nanmin(speed), np.nanmax(speed), np.nanmedian(speed)))
p = dict(profile, dtype="float32", nodata=np.nan, count=1, compress="lzw")
with rasterio.open("pred_results/tsunami_speed.tif", "w", **p) as d:
    d.write(speed.astype("float32"), 1)

# ---------------------------------------------------------------- 4-6. geodesic paths, travel time
to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
to_xy = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
inv = ~transform
rows = []
segments = []
for _, r in paths.iterrows():
    line = r.geometry.geoms[0] if r.geometry.geom_type == "MultiLineString" else r.geometry
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    lo0, la0 = to_ll.transform(x0, y0)
    lo1, la1 = to_ll.transform(x1, y1)
    dist = geod.inv(lo0, la0, lo1, la1)[2]
    n = int(dist // 1000)                                 # Geodetic Densify: ~1 km vertices
    ll = np.array([(lo0, la0)] + geod.npts(lo0, la0, lo1, la1, n) + [(lo1, la1)])
    seg_len = geod.inv(ll[:-1, 0], ll[:-1, 1], ll[1:, 0], ll[1:, 1])[2]
    xs, ys = to_xy.transform(ll[:, 0], ll[:, 1])
    mx, my = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    c, rr = inv * (mx, my)
    c, rr = np.floor(c).astype(int), np.floor(rr).astype(int)
    sp = speed[rr, c]
    ok = np.isfinite(sp)
    t_h = seg_len[ok] / sp[ok] / 3600.0
    rows.append({"Route": r["Route"], "geodesic_km": seg_len.sum() / 1000,
                 "km_with_speed": seg_len[ok].sum() / 1000, "mean_speed_ms": seg_len[ok].sum() / (t_h.sum() * 3600),
                 "TravelTimeHours": t_h.sum()})
    segments.append((r["Route"], xs, ys, np.concatenate([[0], np.cumsum(np.where(ok, seg_len / np.where(ok, sp, 1), 0))]) / 3600))
travel_times = pd.DataFrame(rows)
print(travel_times.round(2).to_string(index=False))
travel_times.to_csv("pred_results/tsunami_travel_times.csv", index=False)

# ---------------------------------------------------------------- arrival-time surfaces
cost = np.where(np.isfinite(speed), cell / np.where(np.isfinite(speed), speed, 1.0), np.inf)
arrival = {}
for _, r in paths.iterrows():
    line = r.geometry.geoms[0] if r.geometry.geom_type == "MultiLineString" else r.geometry
    c, rr = inv * line.coords[0]                          # path start = tsunami origin (offshore)
    mcp = MCP_Geometric(cost)
    cum, _ = mcp.find_costs([(int(rr), int(c))])
    cum[~np.isfinite(cost)] = np.nan
    arrival[r["Route"].split(" to ")[0]] = cum / 3600.0
arrival_la_palma = arrival["La Palma"]
arrival_puerto_rico = arrival["Puerto Rico Trench"]
arrival_currituck = arrival["Currituck Slide"]
vb = sources[sources["Name"] == "Virginia Beach"].geometry.iloc[0]
c, rr = inv * (vb.x, vb.y)
for k, a in arrival.items():
    print(f"least-time arrival at Virginia Beach from {k}: {a[int(rr), int(c)]:.2f} h")
p3 = dict(profile, dtype="float32", nodata=np.nan, count=3, compress="lzw")
with rasterio.open("pred_results/tsunami_arrival_time_hours.tif", "w", **p3) as d:
    for i, (k, a) in enumerate(arrival.items(), start=1):
        d.write(a.astype("float32"), i)
        d.set_band_description(i, k)

# ---------------------------------------------------------------- 7. map
b = rasterio.transform.array_bounds(dem.shape[0], dem.shape[1], transform)
ext = (b[0], b[2], b[1], b[3])
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
ax = axes[0]
im = ax.imshow(speed, extent=ext, cmap="viridis")
fig.colorbar(im, ax=ax, shrink=0.7, label="Tsunami speed (m/s)")
for route, xs, ys, cum_h in segments:
    ax.plot(xs, ys, color="red", lw=1.5)
    ax.annotate(f"{route.split(' to ')[0]}: {cum_h[-1]:.1f} h", (xs[0], ys[0]), color="white", fontsize=9)
ax.set_title("Tsunami speed sqrt(g*d) and travel time along geodesic paths to Virginia Beach")
ax = axes[1]
first_arrival = np.fmin(np.fmin(arrival_la_palma, arrival_puerto_rico), arrival_currituck)
im = ax.imshow(arrival_la_palma, extent=ext, cmap="magma_r", vmax=np.nanpercentile(arrival_la_palma, 99))
cs = ax.contour(np.flipud(arrival_la_palma), levels=range(1, 16), extent=ext, colors="white", linewidths=0.4)
fig.colorbar(im, ax=ax, shrink=0.7, label="Arrival time from La Palma (h)")
ax.plot(vb.x, vb.y, "c*", ms=12)
ax.set_title("Least-time arrival surface, La Palma source")
for a in axes:
    a.set_xlim(ext[0], ext[1]); a.set_ylim(ext[2], ext[3]); a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig("pred_results/tsunami_travel_time.png", dpi=100)
