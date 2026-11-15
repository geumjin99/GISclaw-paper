# Wrong: relationship as plain foreign-key counts (ParentFacilityID.value_counts) without the
# Facilities table, so the campground-campsite relationship is not materialised
import os, pandas as pd, geopandas as gpd
os.makedirs("pred_results", exist_ok=True)
parks = gpd.read_file("dataset/National Parks.gpkg"); cs = gpd.read_file("dataset/Campsites.gpkg")
fac = gpd.read_file("dataset/Facilities.gpkg"); orgs = pd.read_csv("dataset/OrganizationsTable.csv")
attrs = pd.read_csv("dataset/CampsiteAttributes.csv", low_memory=False)
parks.merge(orgs, left_on="ParentOrgID", right_on="OrgID").drop(columns="geometry").to_csv("pred_results/parks.csv", index=False)
cnt = attrs.groupby("EntityID").size().rename("n_related").reset_index()
cs.merge(cnt, left_on="CampsiteID", right_on="EntityID", how="left").drop(columns="geometry").to_csv("pred_results/campsite_relate.csv", index=False)
fk = cs["ParentFacilityID"].value_counts().rename_axis("FacilityID").rename("n_campsites").reset_index()
fk.to_csv("pred_results/campsites_per_campground.csv", index=False)
