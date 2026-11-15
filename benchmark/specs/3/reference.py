"""Reference solution for task 3: burn scars from Landsat 8 (2014 pre-fire, 2015 post-fire), Montana.

NBR = (NIR - SWIR2) / (NIR + SWIR2) = (B5 - B7) / (B5 + B7). The supplied rasters have 8 bands
(no panchromatic band), in Landsat order: 1 coastal, 2 blue, 3 green, 4 red, 5 NIR, 6 SWIR1,
7 SWIR2, 8 cirrus, so NIR = band 5 and SWIR2 = band 7. The edge column holding 65536 is nodata.
dNBR = NBR_2014 - NBR_2015; burned = dNBR > 0.27 (USGS moderate-low severity and above - the
images are uncorrected L1T digital numbers, whose unburned baseline already sits at ~0.05, so
the 0.10 'low severity' break picks up phenology noise). The burned mask is sieved (< 11 px
removed) and vectorised to polygons.
"""
import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

THRESH = 0.27


def nbr(path):
    with rasterio.open(path) as src:
        nir = src.read(5).astype("float64")
        swir2 = src.read(7).astype("float64")
        nod = src.nodata
        transform, crs = src.transform, src.crs
    bad = (nir >= nod) | (swir2 >= nod) if nod is not None else np.zeros(nir.shape, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (nir - swir2) / (nir + swir2)
    v[bad] = np.nan
    return v, transform, crs


def main():
    nbr_2014, transform, crs = nbr("dataset/G_2014.tif")
    nbr_2015, _, _ = nbr("dataset/G_2015.tif")
    dnbr = nbr_2014 - nbr_2015

    burned = np.where(np.isfinite(dnbr), dnbr > THRESH, False).astype("uint8")
    burned = features.sieve(burned, size=11, connectivity=8)
    geoms = [shape(g) for g, v in features.shapes(burned, mask=burned == 1, transform=transform, connectivity=8)]
    burn_scars = gpd.GeoDataFrame({"burned": np.ones(len(geoms), int)}, geometry=geoms, crs=crs)
    burn_scars["area_km2"] = burn_scars.area / 1e6
    print(f"burn-scar polygons: {len(burn_scars)}, total area {burn_scars.area_km2.sum():.1f} km2")

    fig, ax = plt.subplots(figsize=(11, 10))
    l, b, r, t = rasterio.transform.array_bounds(dnbr.shape[0], dnbr.shape[1], transform)
    im = ax.imshow(dnbr, extent=(l, r, b, t), cmap="RdYlGn_r", vmin=-0.3, vmax=0.7)
    plt.colorbar(im, ax=ax, shrink=0.7, label="dNBR (NBR 2014 - NBR 2015)")
    burn_scars.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.legend(handles=[Patch(facecolor="none", edgecolor="black", label=f"Burn scar (dNBR > {THRESH})")],
              loc="lower right")
    ax.set_title("Burn scars 2014-2015 (Landsat 8 dNBR)")
    ax.set_xlabel("Easting (m, UTM 12N)")
    ax.set_ylabel("Northing (m)")
    plt.savefig("pred_results/burn_scar_analysis.png", dpi=150, bbox_inches="tight")
    return dnbr, burn_scars


dnbr, burn_scars = main()
