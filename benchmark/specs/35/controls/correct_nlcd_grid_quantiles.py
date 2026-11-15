"""Correct: works on the NLCD (Albers) grid with quantile burn breaks; the analysis is local to a
function and only the figure is written.
"""
import numpy as np, rasterio
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS

def on_grid(path, band, grid, crs_override=None, nearest=False):
    H, W, T, C = grid
    with rasterio.open(path) as s:
        out = np.full((H, W), np.nan, "float32")
        reproject(s.read(band).astype("float32"), out, src_transform=s.transform, src_crs=crs_override or s.crs,
                  dst_transform=T, dst_crs=C, src_nodata=0 if crs_override else s.nodata, dst_nodata=np.nan,
                  resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return out

def main():
    with rasterio.open("dataset/Sonoma_NLCD2011.tif") as s:
        grid = (s.height, s.width, s.transform, s.crs)
        resx = s.transform.a
    lc = on_grid("dataset/Sonoma_NLCD2011.tif", 1, grid, nearest=True)
    def NBR(p):
        n, w = on_grid(p, 5, grid), on_grid(p, 7, grid)
        return (n - w) / (n + w)
    severity = NBR("dataset/Before_L8.tif") - NBR("dataset/After_L8.tif")
    z = on_grid("dataset/DEM_30m.tif", 1, grid, crs_override=CRS.from_epsg(3857))
    dzdy, dzdx = np.gradient(z, resx, resx)
    slp = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))
    q = np.nanquantile(severity, [0.2, 0.4, 0.6, 0.8])
    b = np.where(np.isnan(severity), np.nan, np.digitize(severity, q) + 1)
    sc = np.where(np.isnan(slp), np.nan, np.digitize(slp, [5, 15, 25, 35]) + 1)
    lcc = np.full(lc.shape, np.nan)
    for (lo, hi), v in {(0, 12): 1, (21, 22): 3, (23, 23): 4, (24, 31): 5, (41, 44): 1, (52, 52): 3,
                        (71, 71): 4, (81, 81): 3, (82, 82): 4, (90, 95): 2}.items():
        lcc[(lc >= lo) & (lc <= hi)] = v
    combined = 0.55 * sc + 0.30 * b + 0.15 * lcc
    fig, ax = plt.subplots(figsize=(8, 8))
    m = ax.imshow(combined, cmap="inferno_r")
    fig.colorbar(m, ax=ax)
    fig.savefig("pred_results/landslide_map.png")

main()
