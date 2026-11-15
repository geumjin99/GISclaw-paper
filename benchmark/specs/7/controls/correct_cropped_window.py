"""Correct: the raster is cropped to the buildings' bounding box (rasterio window) before the
analysis; W = -200 cm, depth in metres, per-building mean depth via zonal statistics on the
cropped grid.
"""
import numpy as np, geopandas as gpd, rasterio, matplotlib.pyplot as plt
from rasterio.windows import from_bounds
from rasterstats import zonal_stats
def analyse():
    bld = gpd.read_file('dataset/StudyAreaBuildings.shp')
    with rasterio.open('dataset/Elevation_2050.tif') as s:
        win = from_bounds(*bld.total_bounds, transform=s.transform).round_offsets().round_lengths()
        z = s.read(1, window=win).astype(float); tr = s.window_transform(win)
    z[z == 0] = np.nan
    depth_m = np.where(z < -200, (-200 - z) / 100.0, np.nan)
    st = zonal_stats(bld, np.nan_to_num(depth_m, nan=-1), affine=tr, nodata=-1, stats=['mean'], all_touched=True)
    bld['depth_m'] = [x['mean'] if x['mean'] is not None else 0.0 for x in st]
    with np.errstate(divide='ignore'):
        loss = (0.298 * np.log(bld.depth_m) + 1.4502) * 271 * bld.Shape_Area
    bld['loss_eur'] = np.where(bld.depth_m > 0, loss.clip(lower=0), 0)
    l, b_, r, t = rasterio.transform.array_bounds(z.shape[0], z.shape[1], tr)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(depth_m, extent=(l, r, b_, t), cmap='Blues')
    bld[bld.loss_eur > 0].plot(ax=ax, column='loss_eur', cmap='Reds', legend=True)
    ax.set_title(f'Flood depth 2050 and building damage: {bld.loss_eur.sum()/1e6:.0f} M')
    fig.savefig('pred_results/flooding_analysis.png', dpi=120)
analyse()
