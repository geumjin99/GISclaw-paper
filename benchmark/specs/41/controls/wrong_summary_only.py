# Wrong: prints summaries of the inputs only; no join, relate or relationship materialised
import os, geopandas as gpd, pandas as pd
os.makedirs("pred_results", exist_ok=True)
for f in ("National Parks", "Campsites", "Facilities"):
    print(f, len(gpd.read_file(f"dataset/{f}.gpkg")))
a = pd.read_csv("dataset/CampsiteAttributes.csv", low_memory=False)
print(a.groupby("EntityID").size().describe())
open("pred_results/notes.txt", "w").write("reviewed tables")
