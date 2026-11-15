# Wrong: joins by row order (Excel rows are alphabetical by abbreviation, the shapefile is not).
import pandas as pd, geopandas as gpd, matplotlib.pyplot as plt
st = gpd.read_file('dataset/USA_States.shp')[['L0USA_Stat', 'L0USA_St_1', 'geometry']]
tab = pd.DataFrame({'State': st['L0USA_St_1'].sort_values().values})
src = gpd.read_file('dataset/USA_States.shp').sort_values('Sheet1__St')   # stand-in for the sheet, alphabetical
tab['Pop13'] = src['Sheet1__Po'].values; tab['Homeless13'] = src['Sheet1__Ho'].values
m = pd.concat([st.reset_index(drop=True), tab[['Pop13', 'Homeless13']]], axis=1)
m['per10k'] = m.Homeless13 / m.Pop13 * 1e4
gpd.GeoDataFrame(m, geometry='geometry', crs=st.crs).plot(column='per10k', legend=True)
plt.savefig('pred_results/homeless_map.png')
