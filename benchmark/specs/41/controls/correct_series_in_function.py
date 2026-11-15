# Correct: results as Series inside a function (captured locals); join by abbreviation; relationship counts keyed by FacilityID
import os, pandas as pd, geopandas as gpd
def main():
    os.makedirs("pred_results", exist_ok=True)
    parks = gpd.read_file("dataset/National Parks.gpkg"); cs = gpd.read_file("dataset/Campsites.gpkg")
    fac = gpd.read_file("dataset/Facilities.gpkg"); orgs = pd.read_csv("dataset/OrganizationsTable.csv")
    attrs = pd.read_csv("dataset/CampsiteAttributes.csv", low_memory=False)
    park_org = parks[["UNIT_CODE", "ParentOrgID"]].merge(orgs[["OrgID", "OrgAbbrevName"]], left_on="ParentOrgID", right_on="OrgID")
    wy_attrs = attrs[attrs["EntityID"].isin(cs["CampsiteID"])]
    related_counts = wy_attrs.groupby("EntityID").size()
    cg_counts = cs[cs["ParentFacilityID"].isin(fac["FacilityID"])].groupby("ParentFacilityID").size()
    open("pred_results/summary.txt", "w").write(f"{park_org}\n{related_counts.describe()}\n{cg_counts}\n")
    return park_org, related_counts, cg_counts
main()
