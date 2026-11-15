# Wrong: no join of homeless_data.xlsx (copies the pre-joined Sheet1__Ho field); maps raw homeless
# counts - California dominates, ND invisible.
import geopandas as gpd, matplotlib.pyplot as plt
s = gpd.read_file('dataset/USA_States.shp')
s['homeless'] = s['Sheet1__Ho']
s.plot(column='homeless', legend=True, cmap='Reds'); plt.savefig('pred_results/homeless_map.png')
