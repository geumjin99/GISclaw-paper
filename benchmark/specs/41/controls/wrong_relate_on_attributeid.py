# Wrong: relate keyed on AttributeID instead of EntityID; join and relationship correct
import os, pandas as pd, geopandas as gpd
os.makedirs("pred_results", exist_ok=True)
parks = gpd.read_file("dataset/National Parks.gpkg"); cs = gpd.read_file("dataset/Campsites.gpkg")
fac = gpd.read_file("dataset/Facilities.gpkg"); orgs = pd.read_csv("dataset/OrganizationsTable.csv")
attrs = pd.read_csv("dataset/CampsiteAttributes.csv", low_memory=False)
parks.merge(orgs, left_on="ParentOrgID", right_on="OrgID").drop(columns="geometry").to_csv("pred_results/parks.csv", index=False)
r = cs.merge(attrs, left_on="CampsiteID", right_on="AttributeID", how="left")
r["n_related"] = r.groupby("CampsiteID")["EntityID"].transform("count")
r.drop(columns="geometry").to_csv("pred_results/relate.csv", index=False)
cs.merge(fac[["FacilityID", "FacilityName"]], left_on="ParentFacilityID", right_on="FacilityID", how="left").drop(columns="geometry").to_csv("pred_results/cs_cg.csv", index=False)
