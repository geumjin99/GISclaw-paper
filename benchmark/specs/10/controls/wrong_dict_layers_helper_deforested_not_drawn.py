# Wrong: layers are built in a loop into a dict and unpacked (as in correct_layers_dict_unpack), the risky
# area comes from a helper whose FIRST argument is the deforested layer (not a pass-through), and the
# deforested layer itself is never drawn (only roads, protected forest, risky area, planned road); the task
# asks for the deforested area on the map.
import geopandas as gpd, matplotlib.pyplot as plt
from shapely.validation import make_valid

def fix(g):
    g = g.copy()
    g["geometry"] = g.geometry.apply(lambda x: make_valid(x) if x is not None and not x.is_valid else x)
    return g, int((~g.geometry.is_valid).sum())

out = {}
for name, fn in [("roads", "roads"), ("deforested", "deforestedArea"), ("protected", "protectedForest"),
                 ("planned", "planned_road")]:
    g, _ = fix(gpd.read_file(f"dataset/{fn}.geojson"))
    out[name] = g.to_crs("ESRI:102033")
roads, deforested, protected, planned = out["roads"], out["deforested"], out["protected"], out["planned"]

def risk(defo, road):
    buf = gpd.GeoDataFrame(geometry=road.buffer(5500), crs=road.crs).dissolve()
    return gpd.overlay(buf, defo[["geometry"]], how="difference")

risky_area = risk(deforested, planned)
risky_area.to_file("pred_results/predictedRiskyArea.geojson")
fig, ax = plt.subplots(figsize=(10, 10))
protected.plot(ax=ax, color="green", alpha=0.6)
roads.plot(ax=ax, color="grey", linewidth=0.2)
risky_area.plot(ax=ax, color="red")
planned.plot(ax=ax, color="black", linewidth=2)
plt.savefig("pred_results/predictedRiskyArea.png", dpi=120)
