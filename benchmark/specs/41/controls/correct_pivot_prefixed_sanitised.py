# Correct: relate as a value pivot whose columns are 'attr_' + sanitised AttributeName, merged onto the
# Wyoming campsite layer; join via map; relationship as campsite counts per campground name
import os, re, pandas as pd, geopandas as gpd
os.makedirs("pred_results", exist_ok=True)
parks = gpd.read_file("dataset/National Parks.gpkg"); cs = gpd.read_file("dataset/Campsites.gpkg")
fac = gpd.read_file("dataset/Facilities.gpkg"); orgs = pd.read_csv("dataset/OrganizationsTable.csv")
attrs = pd.read_csv("dataset/CampsiteAttributes.csv", low_memory=False).dropna(subset=["EntityID"])
attrs["EntityID"] = attrs["EntityID"].astype("int64")
attrs = attrs[attrs["EntityID"].isin(cs["CampsiteID"].astype("int64"))]
wide = attrs.pivot_table(index="EntityID", columns="AttributeName", values="AttributeValue", aggfunc="first")
wide.columns = ["attr_" + re.sub(r"[^0-9A-Za-z]+", "_", c).strip("_") for c in wide.columns]
joined = cs.drop(columns="geometry").merge(wide, left_on="CampsiteID", right_index=True, how="left")
joined.to_csv("pred_results/campsites_joined.csv", index=False)
parks["OrgName"] = parks["ParentOrgID"].map(orgs.set_index("OrgID")["OrgName"])
parks[["UNIT_NAME", "ParentOrgID", "OrgName"]].to_csv("pred_results/parks_org.csv", index=False)
m = fac.merge(cs, left_on="FacilityID", right_on="ParentFacilityID")
per_cg = m.groupby("FacilityName").size().rename("campsites").reset_index()
per_cg.to_csv("pred_results/campsites_per_campground.csv", index=False)
