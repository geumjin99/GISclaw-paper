# Supplemented input files

For ten tasks the data distributed with GeoAnalystBench lack input files that the task
needs. This folder documents which files these are and which public source they come
from. The files themselves are not part of the repository.

`harness/prepare_inputs.py` copies files placed in `benchmark/supplements/<id>/dataset/`
on top of the downloaded data of task `<id>`, under the file names given below.

| Task | Files not in the benchmark's download | Source |
|---|---|---|
| 5, 24 | `High_Blood_Level_Results.shx`, `.prj`; `Sacramento_ZIP_Codes.shx`, `.prj` | The benchmark's own `.shp` files. The index files are rebuilt from them (GDAL, `SHAPE_RESTORE_SHX=YES`). The projection files state the coordinate reference system that the coordinate ranges show: EPSG:3857 for the points and EPSG:4326 for the ZIP code polygons. |
| 23 | `GeorgetownCo_Parcels.gpkg`, `PADUS_CRS_attrib.csv` | Parcels: the parcel layer of the geodatabase in the project package `CRS_Georgetown_Co.ppkx` of the Learn ArcGIS lesson "Assess open space to lower flood insurance cost" (ArcGIS Online item `69648cfd2d0648ef86fa61c5e25d0806`). Attribute table: the rows of The Nature Conservancy's feature layer "Open Space Preservation Community Rating System" (ArcGIS Online item `8a42556edc084af3b274179d2c07fe30`) whose `OBJECTID` is a value of the task's raster `PADUS_CRS_clip_1.tif`. |
| 29 | `AtlanticDEM.tif`, `TsunamiSources.gpkg`, `TsuniamiPaths.gpkg`, `World_Countries.gpkg` | The elevation raster and the three vector layers of the project package `Tsunami.ppkx` of the Learn ArcGIS lesson "Calculate travel time for a tsunami", in the coordinate reference system of the project. |
| 30 | `Neighborhoods`, `Streets`, `Zoning` (shapefiles: `.shp`, `.shx`, `.dbf`, `.prj`, `.cpg`) | The shapefiles `Neighborhoods`, `Street_Network_Database` and `City_of_Seattle_Zoning` of the Esri case study "Designate bike routes for commuting professionals" (ArcGIS Online item `d5ba434ce44b4d68acffc6cc012ad800`, `BikeRoutes.zip`), under the layer names the task uses. |
| 32 | `Roads.geojson` | The file of the same name in the benchmark's data of task 21, which uses the same study area and grid. |
| 38 | `uiuc_drive_network.graphml` | OpenStreetMap: the street network returned by the call in the task's reference solution, `osmnx.graph_from_address('1301 W Green St, Urbana, IL 61801', dist=1000, network_type='drive')`, saved as GraphML. The containers that run the tasks have no network access, so the network is provided as a file. |
| 39 | `G_utm` (shapefile: `.shp`, `.shx`, `.dbf`, `.cpg`, `.sbn`, `.sbx`), `GData_utm.csv` | The example dataset `georgia` of the Python package libpysal, which the task's reference solution loads. |
| 41 | `Campsites.gpkg`, `Facilities.gpkg`, `National Parks.gpkg` | The layers `CampsitesWY`, `FacilitiesWY` and `NationalParks` of the geodatabase `Wyoming Natural Assets.geodatabase` in `Wyoming.zip`, the data of the Learn ArcGIS lesson "Improve campsite data quality using a relationship class" (ArcGIS Online item `70e70857cf3944ce90e9ab9d75d6d6cf`). The polygons of `National Parks.gpkg` are the boundaries of the same park units (same `UNIT_CODE`) in the boundary service of the National Park Service Land Resources Division. |
| 45 | `Elevation.tif` | The elevation raster of the project geodatabase in `Mountain_Lion_Corridors.zip`, the data of the Learn ArcGIS lesson "Find optimal corridors to connect dwindling mountain lion populations". |

For tasks 23, 29, 30 and 41 the benchmark's download contains layer files (`.lyrx`) that
point to these layers without the data; for task 45 it contains the raster's world file
and statistics without the raster; for tasks 38 and 39 it contains no data. The two
shapefiles of tasks 5 and 24 have no attribute table (`.dbf`) in the benchmark's download,
and none is added.

OpenStreetMap data are available under the Open Database License; the other sources are
subject to the terms of their publishers.
