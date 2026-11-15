# Correct: joins homeless_data.xlsx on the abbreviation and maps the 2013 homeless
# COUNT only - the instruction asks for the join and the maps, no rate field.
import geopandas as gpd, matplotlib.pyplot as plt
import zipfile, re, xml.etree.ElementTree as ET, pandas as pd
def read_sheet(path):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    shared = ["".join(t.text or "" for t in si.iter(ns + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    recs = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(ns + "row"):
        d = {}
        for c in r.findall(ns + "c"):
            v = c.find(ns + "v")
            if v is not None:
                d[re.sub(r"\d", "", c.get("r"))] = shared[int(v.text)] if c.get("t") == "s" else float(v.text)
        recs.append(d)
    head = recs[0]
    return pd.DataFrame([{head[k]: r.get(k) for k in head} for r in recs[1:]]).dropna(subset=["State"])
tab = read_sheet("dataset/homeless_data.xlsx")
st = gpd.read_file('dataset/USA_States.shp')[['L0USA_Stat', 'L0USA_St_1', 'geometry']]
j = st.merge(tab, left_on='L0USA_St_1', right_on='State', how='left')
j.to_file('pred_results/states_homeless.gpkg', driver='GPKG')
j.plot(column='Homeless13', cmap='Reds', legend=True, figsize=(12, 6))
plt.savefig('pred_results/homeless_count_2013.png')
