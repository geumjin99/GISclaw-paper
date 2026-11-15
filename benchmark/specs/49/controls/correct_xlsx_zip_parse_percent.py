# Correct: parse the xlsx with zipfile/XML, join on abbreviation, rate in percent, GeoJSON output.
import zipfile, re, xml.etree.ElementTree as ET, pandas as pd, geopandas as gpd, matplotlib.pyplot as plt
ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
z = zipfile.ZipFile('dataset/homeless_data.xlsx')
ss = [''.join(t.text or '' for t in si.iter(ns + 't')) for si in ET.fromstring(z.read('xl/sharedStrings.xml'))]
rows = []
for r in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).iter(ns + 'row'):
    row = {}
    for c in r.findall(ns + 'c'):
        v = c.find(ns + 'v')
        if v is not None:
            row[re.sub(r'\d', '', c.get('r'))] = ss[int(v.text)] if c.get('t') == 's' else float(v.text)
    rows.append(row)
hdr = rows[0]
tab = pd.DataFrame([{hdr[k]: r.get(k) for k in hdr} for r in rows[1:]]).dropna(subset=['State'])
st = gpd.read_file('dataset/USA_States.shp')[['L0USA_Stat', 'L0USA_St_1', 'geometry']]
m = st.merge(tab, left_on='L0USA_St_1', right_on='State')
m['pct_homeless'] = 100 * m.Homeless13 / m.Pop13
m.to_file('pred_results/homeless_join.geojson')
m.plot(column='pct_homeless', legend=True); plt.savefig('pred_results/map.png')
