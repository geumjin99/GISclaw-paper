"""Reference solution for task 11: slope and aspect at coral and sponge observations, Catalina Island.

CatalinaBathymetry.tif: elevation in metres (negative = below sea level), NAD83 / California
Albers (EPSG:3310; rasterio reads the ESRI header as LOCAL_CS), 88.03 m cells, no nodata.
  * slope (degrees) and aspect (degrees clockwise from north, -1 = flat) with Horn's 3x3 method
    (as ArcGIS Slope/Aspect), using the true cell size, on the elevation surface (not on depth:
    flipping the sign turns aspect by 180 degrees);
  * observations (EPSG:4326) are projected to the raster CRS and the slope/aspect of the cell under
    each point are added to the table ("update the database");
  * per VernacularNameCategory: mean slope and mean aspect. Aspect is circular, so the mean
    direction (vector mean of unit vectors) is reported; the arithmetic mean is kept for
    comparison. Flat cells (aspect -1) are excluded from the aspect mean.
  * bar charts of mean slope and mean aspect per category -> pred_results/CoralandSponge.png
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

RASTER_CRS = "EPSG:3310"  # NAD83 / California Albers


def horn(z, cell):
    zp = np.pad(z, 1, mode="edge")
    a, b, c = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    d, f = zp[1:-1, :-2], zp[1:-1, 2:]
    g, h, i = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * cell)   # towards east
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * cell)   # towards south (rows grow southwards)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # aspect = compass direction of steepest descent (ArcGIS convention)
    aspect = np.degrees(np.arctan2(dzdx, -dzdy))   # direction of the uphill gradient...
    aspect = (aspect + 180.0) % 360.0              # ...turned to downhill, 0 = north, clockwise
    aspect[(dzdx == 0) & (dzdy == 0)] = -1.0
    return slope, aspect


def circ_mean(deg):
    deg = deg[deg >= 0]
    if len(deg) == 0:
        return np.nan
    r = np.radians(deg)
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360)


def main():
    with rasterio.open("dataset/CatalinaBathymetry.tif") as src:
        elev = src.read(1).astype("float64")
        transform = src.transform
    slope, aspect = horn(elev, transform.a)

    corals = gpd.read_file("dataset/CoralandSpongeCatalina.geojson")
    pts = corals.to_crs(RASTER_CRS)
    rows, cols = rasterio.transform.rowcol(transform, pts.geometry.x.values, pts.geometry.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < elev.shape[0]) & (cols >= 0) & (cols < elev.shape[1])
    corals["slope"] = np.nan
    corals["aspect"] = np.nan
    corals.loc[inside, "slope"] = slope[rows[inside], cols[inside]]
    corals.loc[inside, "aspect"] = aspect[rows[inside], cols[inside]]
    print(f"{inside.sum()} of {len(corals)} observations on the bathymetry grid")

    grp = corals.groupby("VernacularNameCategory")
    summary = pd.DataFrame({
        "n": grp.size(),
        "mean_slope": grp["slope"].mean(),
        "mean_aspect": grp["aspect"].apply(lambda s: circ_mean(s.dropna().values)),
        "arith_mean_aspect": grp["aspect"].apply(lambda s: s[s >= 0].mean()),
    }).reset_index().sort_values("mean_slope", ascending=False)
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    lab = [f"{c} (n={n})" for c, n in zip(summary.VernacularNameCategory, summary.n)]
    axes[0].barh(lab, summary.mean_slope, color="steelblue")
    axes[0].set_xlabel("Mean slope (degrees)")
    axes[0].set_title("Mean seafloor slope at observations")
    axes[0].invert_yaxis()
    axes[1].barh(lab, summary.mean_aspect, color="darkseagreen")
    axes[1].set_xlim(0, 360)
    axes[1].set_xticks([0, 90, 180, 270, 360], ["N 0", "E 90", "S 180", "W 270", "N 360"])
    axes[1].set_xlabel("Mean aspect (circular mean, degrees from north)")
    axes[1].set_title("Mean aspect at observations")
    axes[1].invert_yaxis()
    plt.suptitle("Coral and sponge categories around Catalina Island: slope and aspect")
    plt.tight_layout()
    plt.savefig("pred_results/CoralandSponge.png", dpi=150)
    return corals, summary


corals, summary = main()
