"""Reference solution for task 27: high-risk groundwater zones that are still undeveloped
(Morrow County, OR)."""
import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

LC_REMAP = {11: 1, 21: 6, 22: 7, 23: 8, 24: 10, 31: 4, 41: 3, 42: 1, 43: 3, 52: 3, 71: 2,
            81: 5, 82: 9, 90: 1, 95: 1}
UNDEVELOPED = ["Deciduous Forest", "Emergent Herbaceous Wetlands", "Hay/Pasture", "Herbaceous",
               "Mixed Forest", "Shrub/Scrub", "Woody Wetlands", "Barren Land"]
NLCD = {11: "Open Water", 21: "Developed, Open Space", 22: "Developed, Low Intensity",
        23: "Developed, Medium Intensity", 24: "Developed, High Intensity", 31: "Barren Land",
        41: "Deciduous Forest", 42: "Evergreen Forest", 43: "Mixed Forest", 52: "Shrub/Scrub",
        71: "Herbaceous", 81: "Hay/Pasture", 82: "Cultivated Crops", 90: "Woody Wetlands",
        95: "Emergent Herbaceous Wetlands"}


def main():
    with rasterio.open("dataset/vulnerable_areas.tif") as v:      # Oregon North, US feet, 98.4 ft
        vul = v.read(1).astype("float64")
        vul[vul == v.nodata] = np.nan
        prof = v.profile.copy()
        vt, vcrs, shape = v.transform, v.crs, (v.height, v.width)
    # land_cover.tif is in Oregon North *metres* on a different, larger grid: align it to the
    # vulnerable-areas grid (nearest neighbour, categorical) before any map algebra.
    with rasterio.open("dataset/land_cover.tif") as l:
        land_cover = np.full(shape, 255, dtype="uint8")
        reproject(l.read(1), land_cover, src_transform=l.transform, src_crs=l.crs, src_nodata=l.nodata,
                  dst_transform=vt, dst_crs=vcrs, dst_nodata=255, resampling=Resampling.nearest)
    print("land cover classes on vulnerable grid:", np.unique(land_cover).tolist())

    # 1. standardise vulnerable areas to 1-10 (min/max of valid cells)
    vmin, vmax = np.nanmin(vul), np.nanmax(vul)
    vul_std = 9 * (vul - vmin) / (vmax - vmin) + 1
    # 2. reclassify land cover (unlisted / NoData -> NoData)
    lc_score = np.full(shape, np.nan)
    for k, s in LC_REMAP.items():
        lc_score[land_cover == k] = s
    # 3. risk zones
    risk_zones = vul_std * 8 + lc_score * 10
    high_risk = risk_zones > 100                         # NaN compares False
    # 4. undeveloped land-cover classes inside the high-risk zones (value = land-cover code)
    undeveloped_codes = [c for c, n in NLCD.items() if n in UNDEVELOPED]
    undeveloped_mask = high_risk & np.isin(land_cover, undeveloped_codes)
    undeveloped_areas = np.where(undeveloped_mask, land_cover, 0).astype("uint8")

    cell_area = abs(vt.a * vt.e) * 0.3048 ** 2           # ft2 -> m2
    print(f"vulnerable min/max {vmin:.0f}/{vmax:.0f}; high-risk cells {int(high_risk.sum())}; "
          f"undeveloped high-risk cells {int(undeveloped_mask.sum())} = {undeveloped_mask.sum() * cell_area / 1e6:.2f} km2")
    for c in undeveloped_codes:
        n = int((undeveloped_areas == c).sum())
        if n:
            print(f"  {c} {NLCD[c]}: {n} cells")

    prof.update(dtype="uint8", nodata=0, count=1, compress="lzw")
    with rasterio.open("pred_results/undeveloped_areas.tif", "w", **prof) as d:
        d.write(undeveloped_areas, 1)
    rp = dict(prof, dtype="float32", nodata=-9999.0)
    with rasterio.open("pred_results/risk_zones.tif", "w", **rp) as d:
        d.write(np.where(np.isfinite(risk_zones), risk_zones, -9999).astype("float32"), 1)


main()
