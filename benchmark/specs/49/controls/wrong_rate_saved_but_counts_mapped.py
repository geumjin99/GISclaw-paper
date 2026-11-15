# Wrong: never joins homeless_data.xlsx (uses the pre-joined Sheet1__* fields). The rate is computed
# and saved, and the map shows the raw homeless counts (natural breaks).
import geopandas as gpd, matplotlib.pyplot as plt
s = gpd.read_file('dataset/USA_States.shp')
s['rate10k'] = 1e4 * s['Sheet1__Ho'] / s['Sheet1__Po']
s['homeless13'] = s['Sheet1__Ho']
s.to_file('pred_results/states_rate.shp')
ax = s.plot(column='homeless13', scheme='naturalbreaks', k=5, cmap='Reds', legend=True, figsize=(10, 6))
plt.savefig('pred_results/homeless_map.png')
