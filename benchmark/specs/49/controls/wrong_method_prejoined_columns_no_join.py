# Wrong: the task prescribes joining homeless_data.xlsx to the states layer; this maps the pre-joined
# Sheet1__* fields of the input shapefile (rate per 100 000, natural-breaks map) and never performs the join.
import geopandas as gpd, matplotlib.pyplot as plt
s = gpd.read_file('dataset/USA_States.shp')
s['homeless_per100k'] = s['Sheet1__Ho'] / s['Sheet1__Po'] * 1e5
ax = s.to_crs(5070).plot(column='homeless_per100k', scheme='NaturalBreaks', k=5, legend=True, cmap='Reds')
plt.savefig('pred_results/homeless_rate.png')
