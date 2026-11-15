# Correct analysis (station-mean IDW), only the presentation is wrong: the grid is computed on an
# mgrid (x index first) and drawn with imshow(z), so the map is rotated/flipped against geography.
# Fails only the kind="figure" placement check.
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson')
pts['x'] = pts.geometry.x; pts['y'] = pts.geometry.y
st = pts.groupby('Station').agg(x=('x', 'mean'), y=('y', 'mean'), do=('MeasureValue', 'mean'))
minx, miny, maxx, maxy = bay.total_bounds
gx, gy = np.mgrid[minx:maxx:200j, miny:maxy:200j]
d2 = (gx[..., None] - st.x.values) ** 2 + (gy[..., None] - st.y.values) ** 2 + 1
w = 1 / d2
do_grid = (w * st['do'].values).sum(-1) / w.sum(-1)
inside = gpd.points_from_xy(gx.ravel(), gy.ravel()).within(bay.geometry.union_all()).reshape(gx.shape)
do_grid[~inside] = np.nan
fig, ax = plt.subplots(figsize=(7, 9))
im = ax.imshow(do_grid, extent=(minx, maxx, miny, maxy), cmap='RdYlBu')   # no .T / origin='lower'
fig.colorbar(im, label='DO (mg/L)')
fig.savefig('pred_results/interploated_water_quality.png')
