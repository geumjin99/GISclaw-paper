# Correct: zone raster via rasterize + bincount; areas in hectares (US survey ft -> m); GeoPackage with renamed key.
import numpy as np, geopandas as gpd, rasterio
from rasterio.features import rasterize

def tabulate():
    pc = gpd.read_file("dataset/parcels.shp")
    with rasterio.open("dataset/impervious.tif") as r:
        cls = r.read(1); tr = r.transform; px_ft2 = abs(tr.a * tr.e)
    zone = rasterize(((g, i + 1) for i, g in enumerate(pc.geometry)), out_shape=cls.shape, transform=tr, fill=0,
                     dtype="int32")
    n = len(pc) + 1
    imp = np.bincount(zone[cls == 0].ravel(), minlength=n)[1:] * px_ft2
    per = np.bincount(zone[cls == 1].ravel(), minlength=n)[1:] * px_ft2
    ft2_to_ha = 0.3048006096 ** 2 / 1e4
    out = pc.assign(IMP_HA=imp * ft2_to_ha, PER_HA=per * ft2_to_ha)
    # GeoJSON cannot carry this non-EPSG State Plane CRS (GDAL writes no crs member): GeoPackage
    # instead, so that the deliverable carries a CRS
    out.to_file("pred_results/parcel_impervious_join.gpkg", driver="GPKG")

tabulate()
