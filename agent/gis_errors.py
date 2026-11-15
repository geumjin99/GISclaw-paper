"""G4 GIS library error knowledge base: generic error -> hint, matched on the last part of
a failed block's error output. Library-level knowledge only (geopandas, shapely, pyproj,
rasterio, fiona), written from the libraries' documented behaviour; nothing per task."""
import re

KB = [
    (r"Cannot transform naive geometries|Please set a crs on the object first",
     "The layer has no CRS. If you know which CRS its coordinates are in, use .set_crs(...) (this does not "
     "move coordinates); use .to_crs(...) only to convert between two known CRS."),
    (r"CRS mismatch between the CRS of left geometries and the CRS of right geometries|"
     r"crs of the (left|right) GeoDataFrame",
     "The two layers use different CRS. Reproject one with .to_crs(other.crs) before the spatial operation."),
    (r"'index_right'.*(already|cannot)|index_right.*exist",
     "sjoin adds a column 'index_right'; drop or rename an existing 'index_right'/'index_left' column first."),
    (r"(TopologyException|GEOSException).*(Self-intersection|side location conflict|invalid)",
     "Some geometries are invalid. Repair them first, e.g. gdf.geometry = gdf.geometry.make_valid() "
     "(or .buffer(0) for polygons)."),
    (r"object has no attribute 'append'",
     "DataFrame.append was removed in pandas 2; use pd.concat([a, b], ignore_index=True)."),
    (r"operands could not be broadcast together with shapes",
     "Two arrays have different shapes. If they are rasters, they are on different grids: resample one onto the "
     "other's grid (rasterio.warp.reproject with the target transform/shape) before combining."),
    (r"(No such file or directory|DriverError).*(\.shp|\.tif|\.geojson|\.gpkg|dataset/)",
     "The path does not exist. List dataset/ to get the exact file names (they are case-sensitive)."),
    (r"not recognized as (a|being in a) supported file format",
     "The file format is not supported by this reader; check the extension, or open it with another library."),
    (r"(CRSError|Invalid projection|PROJ: proj_create)",
     "The CRS string is not valid. Use a form pyproj accepts, e.g. 'EPSG:32633' or a WKT taken from another layer's .crs."),
    (r"Normalized/laundered field name|Value .* of field .* not successfully written|field name.*too long",
     "Shapefile column names are limited to 10 characters and some types; shorten column names or save as GeoPackage/GeoJSON."),
    (r"GeoSeries.*unary_union.*deprecated|has no attribute 'unary_union'",
     "Use .union_all() instead of .unary_union in recent geopandas."),
    (r"(is not a geometry|does not have geometry|You are adding a column named 'geometry')",
     "The active geometry column is missing or wrong; build the layer with gpd.GeoDataFrame(df, geometry=..., crs=...)."),
    (r"WindowError|Window .* outside|not intersect the raster",
     "The window/geometry does not overlap the raster. Check that both use the same CRS before masking or windowed reading."),
    (r"Input shapes do not overlap raster",
     "rasterio.mask: the shapes do not overlap the raster; reproject the shapes to the raster's CRS first."),
    (r"(Singular matrix|LinAlgError)",
     "The matrix is singular: often duplicate points or perfectly collinear variables; remove duplicates or "
     "check the inputs of the model."),
    (r"MemoryError|Unable to allocate",
     "Out of memory for this array. Process it in blocks/windows or use float32; do not change the resolution or sample the data unless the task allows it."),
]
_C = [(re.compile(p, re.I | re.S), h) for p, h in KB]


def hint(stderr_tail):
    t = (stderr_tail or "")[-3000:]
    for rx, h in _C:
        if rx.search(t):
            return h, rx.pattern
    return None, None
