"""Reference solution for task 5: children with elevated blood lead levels, Sacramento -
privacy-preserving hot spots.

Open-source equivalent of ArcGIS Optimized Hot Spot Analysis with
COUNT_INCIDENTS_WITHIN_HEXAGON_POLYGONS and the ZIP codes as bounding polygons:
  * project points (EPSG:3857) and ZIP polygons (EPSG:4326) to UTM 10N (EPSG:32610);
  * hexagonal tessellation (1 km2 hexagons, as in the original workflow) over the ZIP-code area;
  * count incidents per hexagon (zero-count hexagons inside the study area are kept);
  * Getis-Ord Gi* on the counts with a fixed distance band. The scale of analysis follows the
    OHSA rule: peak of incremental spatial autocorrelation (global Moran's I z-score) over
    increasing distance bands; 999 permutations, False Discovery Rate correction; hot/cold spot
    bins at 90/95/99 % confidence;
  * map hexagons only (no individual points) -> pred_results/blood_lead_hotspots.png
    (file name as instructed).
The point file has no attribute table (.dbf missing), so every point is one incident.
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import libpysal
from esda.getisord import G_Local
from esda.moran import Moran
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CRS_M = "EPSG:32610"
HEX_AREA = 1e6  # m2


def hex_grid(bounds, area):
    """Flat-topped regular hexagons of a given area covering bounds."""
    s = np.sqrt(2 * area / (3 * np.sqrt(3)))  # side length
    w, h = 2 * s, np.sqrt(3) * s
    xmin, ymin, xmax, ymax = bounds
    polys = []
    col = 0
    x = xmin
    while x < xmax + w:
        y = ymin - (h / 2 if col % 2 else 0)
        while y < ymax + h:
            polys.append(Polygon([(x + s * np.cos(a), y + s * np.sin(a)) for a in np.radians(np.arange(0, 360, 60))]))
            y += h
        x += 1.5 * s
        col += 1
    return polys


def fdr_threshold(p, alpha):
    ps = np.sort(p)
    m = len(ps)
    ok = ps <= alpha * np.arange(1, m + 1) / m
    return ps[ok].max() if ok.any() else 0.0


def main():
    np.random.seed(0)
    cases = gpd.read_file("dataset/High_Blood_Level_Results.shp").to_crs(CRS_M)
    zips = gpd.read_file("dataset/Sacramento_ZIP_Codes.shp").to_crs(CRS_M)
    study = zips.geometry.union_all()

    hexes = gpd.GeoDataFrame(geometry=hex_grid(zips.total_bounds, HEX_AREA), crs=CRS_M)
    hexes = hexes[hexes.intersects(study)].reset_index(drop=True)
    hexes["hex_id"] = np.arange(len(hexes))
    j = gpd.sjoin(cases, hexes[["hex_id", "geometry"]], how="inner", predicate="within")
    hexes["count"] = hexes["hex_id"].map(j.groupby("hex_id").size()).fillna(0).astype(int)
    print(f"{len(cases)} cases, {len(hexes)} hexagons, {hexes['count'].sum()} cases counted")

    # scale of analysis: incremental spatial autocorrelation
    cent = np.c_[hexes.centroid.x, hexes.centroid.y]
    spacing = np.sqrt(3) * np.sqrt(2 * HEX_AREA / (3 * np.sqrt(3)))  # centre-to-centre distance
    y = hexes["count"].values.astype(float)
    zs = []
    dists = spacing * np.arange(1.05, 6.1, 0.5)
    for d in dists:
        w = libpysal.weights.DistanceBand(cent, threshold=d, binary=True, silence_warnings=True)
        w.transform = "r"
        zs.append(Moran(y, w, permutations=0).z_norm)
    zs = np.array(zs)
    peaks = [i for i in range(1, len(zs) - 1) if zs[i] > zs[i - 1] and zs[i] >= zs[i + 1]]
    band = dists[peaks[0]] if peaks else dists[int(np.argmax(zs))]
    print("incremental Moran z:", dict(zip(np.round(dists).astype(int).tolist(), np.round(zs, 2).tolist())),
          "-> band", round(band))

    w = libpysal.weights.DistanceBand(cent, threshold=band, binary=True, silence_warnings=True)
    gi = G_Local(y, w, star=True, permutations=999, seed=0, n_jobs=1)
    hexes["gi_z"] = gi.Zs
    hexes["gi_p"] = gi.p_sim
    hexes["gi_bin"] = 0
    for conf, alpha in ((1, 0.10), (2, 0.05), (3, 0.01)):
        thr = fdr_threshold(hexes["gi_p"].values, alpha)
        sig = hexes["gi_p"] <= thr
        hexes.loc[sig, "gi_bin"] = np.sign(hexes.loc[sig, "gi_z"]).astype(int) * conf
    print("Gi* bins:", hexes["gi_bin"].value_counts().sort_index().to_dict())

    # map: hot spots (left) and case counts per hexagon (right); no individual points
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    colors = {-3: "#2166ac", -2: "#67a9cf", -1: "#d1e5f0", 0: "#f7f7f7", 1: "#fddbc7", 2: "#ef8a62", 3: "#b2182b"}
    labels = {-3: "Cold 99%", -2: "Cold 95%", -1: "Cold 90%", 0: "Not significant", 1: "Hot 90%",
              2: "Hot 95%", 3: "Hot 99%"}
    ax = axes[0]
    hexes.plot(ax=ax, color=hexes["gi_bin"].map(colors), edgecolor="grey", linewidth=0.2)
    zips.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.legend(handles=[Patch(color=colors[k], label=labels[k]) for k in sorted(colors)
                       if (hexes["gi_bin"] == k).any()], loc="lower left")
    ax.set_title(f"Optimized hot spot analysis (Gi*, FDR, band {band / 1000:.1f} km)")
    ax.set_axis_off()
    ax = axes[1]
    hexes.plot(ax=ax, column="count", cmap="Reds", edgecolor="grey", linewidth=0.2, legend=True,
               legend_kwds={"label": "Children with elevated blood lead (count per 1 km2 hexagon)", "shrink": 0.7})
    zips.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_title("Cases summarised within hexagons")
    ax.set_axis_off()
    plt.suptitle("Elevated blood lead levels in children, Sacramento (aggregated for privacy)")
    plt.savefig("pred_results/blood_lead_hotspots.png", dpi=150, bbox_inches="tight")
    return hexes


hexes = main()
