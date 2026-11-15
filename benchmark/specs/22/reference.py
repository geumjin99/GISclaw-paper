"""Reference solution for task 22: mountain lion corridor, weighted composite cost surface (map algebra)."""
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WEIGHTS = {"distance": 1.0, "ruggedness": 1.25, "protected_status_reclassified": 1.0,
           "landcover_reclassified": 1.25}


def read(name):
    with rasterio.open(f"dataset/{name}.tif") as src:
        return src.read(1).astype("float64"), src.profile


def main():
    layers, profile = {}, None
    for name in WEIGHTS:
        layers[name], profile = read(name)
    shapes = {a.shape for a in layers.values()}
    assert len(shapes) == 1, shapes            # map algebra needs identical grids

    # landcover_reclassified is a 0-10 score; 255 is the uint8 NoData code (ocean, SW corner)
    # that lost its NoData flag in the GeoTIFF - mask it instead of treating it as cost 255.
    nodata = layers["landcover_reclassified"] == 255
    for a in layers.values():
        a[nodata] = np.nan

    # Raster Calculator: weighted sum, divided by the weight total so the result stays on the
    # 0-10 cost scale of the inputs (the division does not change the ranking of cells).
    wsum = sum(WEIGHTS.values())
    cost_surface = sum(w * layers[n] for n, w in WEIGHTS.items()) / wsum
    print("cost surface: min %.2f max %.2f mean %.3f, NoData cells %d"
          % (np.nanmin(cost_surface), np.nanmax(cost_surface), np.nanmean(cost_surface), nodata.sum()))

    b = rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"])
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cost_surface, cmap="viridis", extent=(b[0], b[2], b[1], b[3]))
    fig.colorbar(im, ax=ax, shrink=0.8, label="Weighted cost (0-10)")
    ax.set_title("Mountain lion corridor - composite cost surface\n"
                 "(distance 1, ruggedness 1.25, protected status 1, land cover 1.25)")
    ax.set_xlabel("Easting (m, NAD83 / UTM 11N)")
    ax.set_ylabel("Northing (m)")
    fig.savefig("pred_results/mountainLionCorridor.png", dpi=120, bbox_inches="tight")


main()
