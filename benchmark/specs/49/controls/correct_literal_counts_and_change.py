# Correct: literal reading - join on abbreviation, calculated field = % change 2012-13 ("trends"), maps of 2013 count and change.
import zipfile, re, xml.etree.ElementTree as ET, pandas as pd, geopandas as gpd, matplotlib.pyplot as plt

def read_sheet(path):
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(path)
    shared = [''.join(t.text or '' for t in si.iter(ns + 't')) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
    recs = []
    for r in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).iter(ns + 'row'):
        d = {}
        for c in r.findall(ns + 'c'):
            v = c.find(ns + 'v')
            if v is not None:
                d[re.sub(r'\d', '', c.get('r'))] = shared[int(v.text)] if c.get('t') == 's' else float(v.text)
        recs.append(d)
    head = recs[0]
    return pd.DataFrame([{head[k]: r.get(k) for k in head} for r in recs[1:]]).dropna(subset=['State'])

tab = read_sheet("dataset/homeless_data.xlsx")
st = gpd.read_file("dataset/USA_States.shp")[["L0USA_Stat", "L0USA_St_1", "geometry"]]
joined = st.merge(tab, left_on="L0USA_St_1", right_on="State", how="left")
joined["ChangePct"] = joined["Change"] * 100
joined.to_file("pred_results/homeless_join.gpkg", driver="GPKG")
fig, ax = plt.subplots(1, 2, figsize=(18, 6))
joined.plot(column="Homeless13", cmap="Reds", legend=True, ax=ax[0]); ax[0].set_title("Homeless 2013")
joined.plot(column="ChangePct", cmap="RdBu_r", legend=True, ax=ax[1]); ax[1].set_title("Change 2012-13 (%)")
for a in ax:
    a.set_xlim(-2.1e7, -7e6)
plt.savefig("pred_results/homeless_maps.png")
