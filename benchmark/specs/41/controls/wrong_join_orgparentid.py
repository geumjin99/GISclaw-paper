# Wrong: park join on OrgParentID (as in the benchmark's CodeString) -> wrong organisation; other parts correct
import os, pandas as pd, geopandas as gpd
os.makedirs("pred_results", exist_ok=True)
parks = gpd.read_file("dataset/National Parks.gpkg"); cs = gpd.read_file("dataset/Campsites.gpkg")
fac = gpd.read_file("dataset/Facilities.gpkg"); orgs = pd.read_csv("dataset/OrganizationsTable.csv")
attrs = pd.read_csv("dataset/CampsiteAttributes.csv", low_memory=False)
pj = parks.merge(orgs, left_on="ParentOrgID", right_on="OrgParentID", how="left")
pj.drop(columns="geometry").to_csv("pred_results/parks_joined.csv", index=False)
cnt = attrs.groupby("EntityID").size().rename("n_related").reset_index()
cs.merge(cnt, left_on="CampsiteID", right_on="EntityID", how="left").drop(columns="geometry").to_csv("pred_results/campsite_relate.csv", index=False)
cs.merge(fac[["FacilityID", "FacilityName"]], left_on="ParentFacilityID", right_on="FacilityID", how="left").drop(columns="geometry").to_csv("pred_results/cs_cg.csv", index=False)
