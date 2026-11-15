# Correct: SAVI on raw digital numbers, the before image warped onto the after grid, gains dropped,
# per-field means with rasterstats.
import os, shutil, numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling
from rasterstats import zonal_stats
shutil.copy('dataset/Field_boundaries.shp', '/tmp/fb.shp'); os.chmod('/tmp/fb.shp', 0o644)
os.environ['SHAPE_RESTORE_SHX'] = 'YES'
fields = gpd.read_file('/tmp/fb.shp').set_crs(3857)            # coordinates are Web Mercator
a = rasterio.open('dataset/After_storm.tif'); b = rasterio.open('dataset/Before_storm.tif')
fields = fields.to_crs(a.crs)
def savi(red, nir):                                              # authors' DN-based SAVI
    red = red.astype('float32'); nir = nir.astype('float32')
    s = (nir - red) / (nir + red + 0.5) * 1.5
    s[(red == 0) | (nir == 0)] = np.nan
    return s
savi_after = savi(a.read(3), a.read(4))
red_b = np.zeros((a.height, a.width), 'uint16'); nir_b = np.zeros_like(red_b)
for band, out in ((3, red_b), (4, nir_b)):                       # put Before on the After grid
    reproject(rasterio.band(b, band), out, dst_transform=a.transform, dst_crs=a.crs, resampling=Resampling.nearest)
loss = savi(red_b, nir_b) - savi_after
del red_b, nir_b, savi_after
loss[loss < 0] = np.nan                                          # remap: drop gains
zs = zonal_stats(fields, loss, affine=a.transform, stats=['mean'], nodata=np.nan)
del loss
fields['veg_loss'] = [z['mean'] for z in zs]
fields.to_file('pred_results/fields_loss.gpkg')
fields.plot(column='veg_loss', legend=True); plt.savefig('pred_results/loss.png')
