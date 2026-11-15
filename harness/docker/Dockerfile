# gisclaw-exec: isolated execution image for agent code.
#   docker build -t gisclaw-exec:1.0 harness/docker
FROM python:3.13-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg \
    MPLCONFIGDIR=/opt/mplconfig CARTOPY_DATA_DIR=/opt/cartopy
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libexpat1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-exec.txt /opt/requirements-exec.txt
COPY constraints.txt /opt/constraints.txt
RUN pip install -r /opt/requirements-exec.txt -c /opt/constraints.txt
# Natural Earth layers that cartopy would otherwise download at run time (containers have
# no network): fetched once here with cartopy's own downloader into CARTOPY_DATA_DIR.
RUN python -c "\
import cartopy; \
from cartopy.io import shapereader; \
cartopy.config['data_dir'] = '/opt/cartopy'; \
scales = ('110m', '50m', '10m'); \
layers = [('physical', n, s) for n in ('coastline', 'land', 'ocean', 'lakes', 'rivers_lake_centerlines') for s in scales]; \
layers += [('cultural', n, s) for n in ('admin_0_boundary_lines_land', 'admin_1_states_provinces_lakes') for s in scales]; \
layers += [('cultural', 'admin_1_states_provinces_lines', '50m'), ('cultural', 'admin_0_countries', '10m')]; \
[shapereader.natural_earth(resolution=s, category=c, name=n) for c, n, s in layers]" \
    && chmod -R a+rX /opt/cartopy
# warm matplotlib font cache so the first figure is not slow; make it world-readable
RUN mkdir -p /opt/mplconfig && python -c "import matplotlib.pyplot as plt; plt.figure()" \
    && chmod -R a+rwX /opt/mplconfig && pip freeze > /opt/pip-freeze.txt
# fail the build if any preloaded or commonly used library cannot be imported
RUN python -c "import numpy, pandas, geopandas, rasterio, fiona, pyproj, shapely, scipy, sklearn, skimage, matplotlib, rasterstats, libpysal, esda, mgwr, networkx, osmnx, xarray, netCDF4, pykrige, statsmodels, cartopy, contextily, h3, xgboost, lightgbm, iris, geoplot, splot, nltk, rioxarray, spaghetti; print(rasterio.__gdal_version__, pyproj.proj_version_str)"
# coastlines must render from the files fetched above
RUN python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, cartopy.crs as ccrs; ax=plt.axes(projection=ccrs.PlateCarree()); ax.coastlines(); plt.savefig('/tmp/c.png')"
WORKDIR /task
