"""Correct: the analysis is right and includes the instructed network step (networkx shortest paths to
downtown); the map shows only the service area and neighbourhoods, without the proposed routes.
The map checks are optional because the instruction asks for no map.
"""
import geopandas as gpd, pandas as pd, matplotlib.pyplot as plt
nb = gpd.read_file("dataset/Neighborhoods.shp"); st = gpd.read_file("dataset/Streets.shp")
ch = nb[nb.L_HOOD == "CAPITOL HILL"].dissolve()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
near = art[art.intersects(ch.geometry.iloc[0].buffer(5))]
names = ["MADISON ST", "E MADISON ST", "10TH AVE E", "BROADWAY", "BROADWAY E", "23RD AVE", "23RD AVE E", "24TH AVE E", "TURNER WAY E"]
routes = near[near.ORD_STNAME.isin(names)]
service = gpd.GeoDataFrame(geometry=[routes.buffer(2640).union_all()], crs=st.crs)
inside = service.intersection(ch.geometry.iloc[0]).area.sum()
stats = {'buffer_ft': 2640.0, 'hood_sqmi': ch.area.sum() / 5280 ** 2, 'served_sqmi': inside / 5280 ** 2,
         'served_share': inside / ch.area.sum(), 'route_mi': routes.length.sum() / 5280}
pd.DataFrame([stats]).to_csv('pred_results/route_stats.csv', index=False)
service.to_file('pred_results/service_area_half_mile.gpkg')
routes.to_file('pred_results/proposed_routes.gpkg')
ax = service.plot(alpha=.3); ch.boundary.plot(ax=ax, color='k'); nb.boundary.plot(ax=ax, color='grey', lw=.3)
ax.set_title('Half-mile service area: %.1f%% of Capitol Hill' % (100 * stats['served_share']))
plt.savefig('pred_results/bike_routes.png')
# Network Analyst step: arterial graph, shortest network path from each corridor to downtown
import networkx as nx
G = nx.Graph()
for geom in art.geometry:
    for part in getattr(geom, "geoms", [geom]):
        c = list(part.coords)
        a, b = (round(c[0][0]), round(c[0][1])), (round(c[-1][0]), round(c[-1][1]))
        G.add_edge(a, b, weight=part.length)
dt = nb[nb.L_HOOD == "DOWNTOWN"].union_all()
dt_nodes = [n for n in G.nodes if dt.distance(gpd.points_from_xy([n[0]], [n[1]])[0]) < 5]
G.add_edges_from((("DT", n) for n in dt_nodes), weight=0.0)
for nm in names:
    seg = routes[routes.ORD_STNAME == nm]
    if seg.empty:
        continue
    c = list(getattr(seg.geometry.iloc[0], "geoms", [seg.geometry.iloc[0]])[0].coords)
    src = (round(c[0][0]), round(c[0][1]))
    try:
        d = nx.shortest_path_length(G, src, "DT", weight="weight")
        print(f"{nm}: network distance to downtown {d / 5280:.2f} mi")
    except nx.NetworkXNoPath:
        print(f"{nm}: not connected to downtown on the arterial network")
