# Wrong: campground assigned by nearest facility (spatial) instead of the ParentFacilityID key; join and relate correct
import os, pandas as pd, geopandas as gpd
os.makedirs("pred_results", exist_ok=True)
parks = gpd.read_file("dataset/National Parks.gpkg"); cs = gpd.read_file("dataset/Campsites.gpkg")
fac = gpd.read_file("dataset/Facilities.gpkg"); orgs = pd.read_csv("dataset/OrganizationsTable.csv")
attrs = pd.read_csv("dataset/CampsiteAttributes.csv", low_memory=False)
parks.merge(orgs, left_on="ParentOrgID", right_on="OrgID").drop(columns="geometry").to_csv("pred_results/parks.csv", index=False)
cnt = attrs.groupby("EntityID").size().rename("n_related").reset_index()
cs.merge(cnt, left_on="CampsiteID", right_on="EntityID", how="left").drop(columns="geometry").to_csv("pred_results/campsite_relate.csv", index=False)
cg = fac[fac["FacilityTypeDescription"] == "Campground"]
near = gpd.sjoin_nearest(cs.to_crs(26912), cg[["FacilityID", "FacilityName", "geometry"]].to_crs(26912))
near.drop(columns="geometry").to_csv("pred_results/campsite_campground.csv", index=False)
near.groupby("FacilityName").size().rename("n").reset_index().to_csv("pred_results/per_campground.csv", index=False)
