# Correct: rioxarray, half-resolution averaged read, post-minus-pre convention, threshold -0.25,
# polygons dissolved and reprojected to WGS84; all inside a function.
import numpy as np, rioxarray, geopandas as gpd, matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio import features
from shapely.geometry import shape

def analyse():
    imgs = {}
    for yr in ('2014', '2015'):
        da = rioxarray.open_rasterio(f'dataset/G_{yr}.tif', masked=True).astype('float64')
        da = da.rio.reproject(da.rio.crs, shape=(da.shape[1] // 2, da.shape[2] // 2), resampling=Resampling.average)
        nir, sw2 = da.sel(band=5), da.sel(band=7)
        imgs[yr] = (nir - sw2) / (nir + sw2)
    change = imgs['2015'] - imgs['2014']          # negative = vegetation loss
    burnt = (change < -0.25).fillna(False).values.astype('uint8')
    tr = change.rio.transform()
    polys = [shape(g) for g, v in features.shapes(burnt, mask=burnt == 1, transform=tr)]
    scars = gpd.GeoDataFrame({'id': range(len(polys))}, geometry=polys, crs=change.rio.crs)
    scars = scars[scars.area > 4 * 3600].to_crs(4326)
    fig, ax = plt.subplots(figsize=(10, 9))
    change.plot(ax=ax, cmap='RdYlGn', vmin=-0.7, vmax=0.3)
    scars.to_crs(change.rio.crs).boundary.plot(ax=ax, color='k', linewidth=0.5)
    ax.set_title('Burn scars (NBR2015 - NBR2014 < -0.25)')
    fig.savefig('pred_results/burn_scar_analysis.png', dpi=130)
    return scars

analyse()
