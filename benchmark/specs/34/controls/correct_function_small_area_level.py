# Correct: inside a function; areal weighting at the small-area level (fraction of each clipped small area inside
# the buffer), municipal shares aggregated from it, choropleth saved inside the function
import geopandas as gpd, pandas as pd, matplotlib.pyplot as plt
def analyse():
    metro = gpd.read_file('dataset/ShikokuMetropolitan.geojson').to_crs(32653)
    roads = gpd.read_file('dataset/AllSeasonRoads.geojson').to_crs(32653)
    pop = gpd.read_file('dataset/ShikokuPopulation.geojson').to_crs(32653)
    pop['full_area'] = pop.area
    rural = metro.loc[metro.AREATYPE == 'RURAL', ['JISCODE', 'AREANAME', 'geometry']].rename(columns={'JISCODE': 'muni_code'})
    pieces = gpd.overlay(pop[['ID', 'D0001', 'full_area', 'geometry']], rural, how='intersection')
    pieces['residents'] = pieces.D0001 * pieces.area / pieces.full_area
    zone = roads.buffer(2000).union_all()
    pieces['frac_in'] = pieces.geometry.intersection(zone).area / pieces.area
    pieces['served'] = pieces.residents * pieces.frac_in
    muni = pieces.groupby('muni_code')[['residents', 'served']].sum()
    muni['access_pct'] = 100 * muni.served / muni.residents
    rmap = rural.merge(muni.reset_index(), on='muni_code')
    overall = 100 * muni.served.sum() / muni.residents.sum()
    fig, ax = plt.subplots(figsize=(10, 7))
    rmap.plot(column='access_pct', cmap='YlGn', legend=True, ax=ax, edgecolor='grey', linewidth=.3)
    ax.set_title(f'Rural population within 2 km of all-season roads: {overall:.1f}% overall')
    fig.savefig('pred_results/accessibility.png')
analyse()
