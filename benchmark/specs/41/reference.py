"""Reference solution for task 41: campsite data quality, Wyoming (joins, relates, relationship class).

Inputs: National Parks,
Campsites and Facilities as GeoPackages from the lesson's geodatabase, next to the official
.lyrx files and CSV tables. Keys as defined in the .lyrx files and the lesson's CodeString:
  1. join   National Parks.ParentOrgID -> OrganizationsTable.OrgID  (park-level organisation;
            the CodeString's OrgParentID is the parent's parent and gives the wrong organisation);
  2. relate Campsites.CampsiteID (1) -> CampsiteAttributes.EntityID (many);
  3. relationship class Facilities.FacilityID (1, origin, campgrounds) -> Campsites.ParentFacilityID
     (many), rule: one campground has 1..100 campsites.
A relationship class is a geodatabase object with no open-source equivalent; it is materialised
here as the one-to-many link table plus per-campground counts and the rule check.
"""
import json
import os

import geopandas as gpd
import pandas as pd

os.makedirs("pred_results", exist_ok=True)

parks = gpd.read_file("dataset/National Parks.gpkg")
campsites = gpd.read_file("dataset/Campsites.gpkg")
facilities = gpd.read_file("dataset/Facilities.gpkg")
orgs = pd.read_csv("dataset/OrganizationsTable.csv", encoding="utf-8-sig")
attrs = pd.read_csv("dataset/CampsiteAttributes.csv", encoding="utf-8-sig")

# 1. join parks -> organisations
parks_org = parks.merge(orgs, left_on="ParentOrgID", right_on="OrgID", how="left", validate="m:1")
parks_org.drop(columns="geometry").to_csv("pred_results/national_parks_organizations.csv", index=False)

# 2. relate campsites -> attributes (one-to-many), materialised per campsite
attrs = attrs.dropna(subset=["EntityID"])
attrs = attrs[attrs["EntityType"] == "Campsite"].copy()
attrs["EntityID"] = attrs["EntityID"].astype("int64")
valued = attrs[attrs["AttributeValue"].notna()].groupby("EntityID")["AttributeName"].nunique()
campsite_relate = (attrs.groupby("EntityID")
                   .agg(n_attributes=("AttributeID", "size"),
                        n_attribute_names=("AttributeName", "nunique"))
                   .reset_index().rename(columns={"EntityID": "CampsiteID"}))
campsite_relate["n_valued_attribute_names"] = campsite_relate["CampsiteID"].map(valued).fillna(0).astype(int)
campsite_relate["in_wyoming"] = campsite_relate["CampsiteID"].isin(campsites["CampsiteID"])
wy_relate = campsites[["CampsiteID", "CampsiteName"]].merge(campsite_relate.drop(columns="in_wyoming"),
                                                            on="CampsiteID", how="left")
for c in ("n_attributes", "n_attribute_names", "n_valued_attribute_names"):
    wy_relate[c] = wy_relate[c].fillna(0).astype(int)
wy_relate.to_csv("pred_results/campsite_attribute_relate.csv", index=False)

# 3. relationship class campground (FacilityID) 1 -> many campsites (ParentFacilityID)
campsite_campground = campsites[["CampsiteID", "CampsiteName", "ParentFacilityID"]].merge(
    facilities[["FacilityID", "FacilityName", "FacilityTypeDescription"]],
    left_on="ParentFacilityID", right_on="FacilityID", how="left")
campsite_campground.to_csv("pred_results/campsite_campground_relationship.csv", index=False)
campgrounds = facilities[facilities["FacilityTypeDescription"] == "Campground"]
campground_campsites = (campgrounds[["FacilityID", "FacilityName"]]
                        .merge(campsites.groupby("ParentFacilityID").size().rename("n_campsites"),
                               left_on="FacilityID", right_index=True, how="left"))
campground_campsites["n_campsites"] = campground_campsites["n_campsites"].fillna(0).astype(int)
campground_campsites["rule_1_to_100_ok"] = campground_campsites["n_campsites"].between(1, 100)
campground_campsites.to_csv("pred_results/campground_campsite_counts.csv", index=False)
orphans = campsites[~campsites["ParentFacilityID"].isin(facilities["FacilityID"])]

summary = {
    "parks_organization": parks_org[["UNIT_NAME", "OrgID", "OrgName"]].to_dict("records"),
    "campsites": int(len(campsites)),
    "campsites_with_attribute_records": int((wy_relate["n_attributes"] > 0).sum()),
    "attribute_records_of_wyoming_campsites": int(wy_relate["n_attributes"].sum()),
    "campgrounds": int(len(campgrounds)),
    "campgrounds_violating_rule_1_to_100": campground_campsites.loc[~campground_campsites["rule_1_to_100_ok"], "FacilityName"].tolist(),
    "campsites_without_parent_facility": int(len(orphans)),
    "missing_parent_facility_ids": sorted(int(x) for x in orphans["ParentFacilityID"].unique()),
}
json.dump(summary, open("pred_results/data_quality_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
