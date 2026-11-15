# Correct: analysis inside a function; the reduction is documented only in a markdown note
import geopandas as gpd
def run():
    p = gpd.read_file("dataset/High_Blood_Level_Results.shp").to_crs(4326)
    lon, lat = p.geometry.x.round(2), p.geometry.y.round(2)
    out = gpd.GeoDataFrame({"lon_2dp": lon, "lat_2dp": lat}, geometry=gpd.points_from_xy(lon, lat), crs=4326)
    before = len(set(zip(p.geometry.x, p.geometry.y)))
    after = len(set(zip(lon, lat)))
    out.to_file("pred_results/deidentified_points.geojson", driver="GeoJSON")
    with open("pred_results/README.md", "w") as f:
        f.write(f"# De-identification\nUnique locations went from {before} to {after}; "
                f"{100 * (before - after) / before:.1f} percent fewer.\n")
run()
