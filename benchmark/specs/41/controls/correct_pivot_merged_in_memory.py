# Correct: value pivot reset_index and merged with campsite columns, kept only in memory; relationship per campground by name
import os, pandas as pd, geopandas as gpd
os.makedirs("pred_results", exist_ok=True)
parks = gpd.read_file("dataset/National Parks.gpkg"); cs = gpd.read_file("dataset/Campsites.gpkg")
fac = gpd.read_file("dataset/Facilities.gpkg"); orgs = pd.read_csv("dataset/OrganizationsTable.csv")
attrs = pd.read_csv("dataset/CampsiteAttributes.csv", low_memory=False)
parks.merge(orgs, left_on="ParentOrgID", right_on="OrgID").drop(columns="geometry").to_csv("pred_results/parks.csv", index=False)
wy = attrs[attrs["EntityID"].isin(cs["CampsiteID"])]
pivot = wy.pivot_table(index="EntityID", columns="AttributeName", values="AttributeValue", aggfunc="first").reset_index()
campsite_pivot = cs[["CampsiteID", "CampsiteName", "Loop"]].merge(pivot, left_on="CampsiteID", right_on="EntityID", how="left")
per_cg = cs.merge(fac, left_on="ParentFacilityID", right_on="FacilityID").groupby("FacilityName").size()
print(campsite_pivot.shape, per_cg)
