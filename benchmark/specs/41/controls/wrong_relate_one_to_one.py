# Wrong: relate collapsed to a one-to-one join (first attribute record only); join and relationship correct
import os, pandas as pd, geopandas as gpd
os.makedirs("pred_results", exist_ok=True)
parks = gpd.read_file("dataset/National Parks.gpkg"); cs = gpd.read_file("dataset/Campsites.gpkg")
fac = gpd.read_file("dataset/Facilities.gpkg"); orgs = pd.read_csv("dataset/OrganizationsTable.csv")
attrs = pd.read_csv("dataset/CampsiteAttributes.csv", low_memory=False)
parks.merge(orgs, left_on="ParentOrgID", right_on="OrgID").drop(columns="geometry").to_csv("pred_results/parks.csv", index=False)
j = cs.merge(attrs.drop_duplicates("EntityID"), left_on="CampsiteID", right_on="EntityID", how="left")
j["n_related"] = j.groupby("CampsiteID")["AttributeID"].transform("size")
j.drop(columns="geometry").to_csv("pred_results/campsites_joined.csv", index=False)
cs.merge(fac[["FacilityID", "FacilityName"]], left_on="ParentFacilityID", right_on="FacilityID", how="left").drop(columns="geometry").to_csv("pred_results/cs_cg.csv", index=False)
