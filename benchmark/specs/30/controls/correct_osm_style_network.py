"""Correct: selects the corridors by street-name patterns with str.contains, builds the half-mile
service area with a dissolved buffer per route (no union step), clips with gpd.clip and reports
the coverage as a percentage variable; network connectivity with networkx.
"""
import geopandas as gpd, networkx as nx, matplotlib.pyplot as plt
nb = gpd.read_file('dataset/Neighborhoods.shp'); st = gpd.read_file('dataset/Streets.shp')
ch = nb[nb.L_HOOD == 'CAPITOL HILL'].dissolve()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
near = art[art.intersects(ch.geometry.iloc[0].buffer(5))]
pat = r'^(E )?MADISON ST$|^10TH AVE E$|^BROADWAY( E)?$|^23RD AVE( E)?$|^24TH AVE E$|^TURNER WAY E$'
routes = near[near.ORD_STNAME.str.contains(pat, regex=True, na=False)]
G = nx.Graph()
for g in art.geometry:
    for p in getattr(g, 'geoms', [g]):
        G.add_edge(tuple(map(round, p.coords[0])), tuple(map(round, p.coords[-1])), w=p.length)
largest = max(nx.connected_components(G), key=len)
on_main_network = sum(tuple(map(round, (g.geoms[0] if hasattr(g, 'geoms') else g).coords[0])) in largest for g in routes.geometry)
service = gpd.clip(gpd.GeoDataFrame(geometry=routes.buffer(2640), crs=st.crs).dissolve(), ch)
coverage_pct = 100 * service.area.sum() / ch.area.sum()
routes.to_file('pred_results/bike_routes.shp')
ax = ch.plot(color='lightyellow'); service.plot(ax=ax, alpha=.4); routes.plot(ax=ax, color='red')
ax.set_title(f'{coverage_pct:.1f}% within 1/2 mile'); plt.savefig('pred_results/bike_plan.png')
