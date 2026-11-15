# Correct: ordinary kriging (pykrige) of station means on a 1 km grid over a padded extent, masked to the Bay
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from shapely import contains_xy
def run():
    pts = gpd.read_file('dataset/DissolvedO2.geojson'); bay = gpd.read_file('dataset/Bay.geojson')
    pts['x'] = pts.geometry.x; pts['y'] = pts.geometry.y
    st = pts.groupby('Station')[['x', 'y', 'MeasureValue']].mean()
    minx, miny, maxx, maxy = bay.total_bounds
    pad = 15000
    gx = np.arange(minx - pad, maxx + pad, 1000.0); gy = np.arange(miny - pad, maxy + pad, 1000.0)
    ok = OrdinaryKriging(st.x.values, st.y.values, st.MeasureValue.values, variogram_model='spherical')
    z, ss = ok.execute('grid', gx, gy)
    z = np.asarray(z)
    GX, GY = np.meshgrid(gx, gy)
    z[~contains_xy(bay.geometry.union_all(), GX, GY)] = np.nan
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(z, origin='lower', extent=(gx[0], gx[-1], gy[0], gy[-1]), cmap='RdYlBu')
    bay.boundary.plot(ax=ax, color='k'); fig.colorbar(im, label='DO mg/L')
    fig.savefig('pred_results/interploated_water_quality.png')
run()
