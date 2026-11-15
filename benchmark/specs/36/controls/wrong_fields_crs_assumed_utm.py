# Wrong: the field coordinates (Web Mercator) are assumed to be in the raster CRS (UTM 12N), so no field
# overlaps the imagery and every field mean is empty.
import os, shutil, numpy as np, rasterio, geopandas as gpd, matplotlib.pyplot as plt
from rasterio.windows import Window
from rasterstats import zonal_stats
shutil.copy('dataset/Field_boundaries.shp', '/tmp/fb.shp'); os.chmod('/tmp/fb.shp', 0o644)
os.environ['SHAPE_RESTORE_SHX'] = 'YES'
a = rasterio.open('dataset/After_storm.tif'); b = rasterio.open('dataset/Before_storm.tif')
fields = gpd.read_file('/tmp/fb.shp').set_crs(a.crs)             # wrong: assumes raster CRS
win = b.window(*a.bounds).round_offsets().round_lengths()
def savi(src, w=None):
    r = src.read(3, window=w).astype('float32') / 1e4; n = src.read(4, window=w).astype('float32') / 1e4
    s = (n - r) / (n + r + 0.5) * 1.5; s[r == 0] = np.nan; return s
loss = savi(b, win)[:a.height, :a.width] - savi(a)
zs = zonal_stats(fields, loss, affine=a.transform, stats=['mean'], nodata=np.nan)
del loss
fields['veg_loss'] = [z['mean'] for z in zs]
fields.to_file('pred_results/fields_loss.gpkg')
fields.boundary.plot(); plt.savefig('pred_results/loss.png')   # all field means are empty
