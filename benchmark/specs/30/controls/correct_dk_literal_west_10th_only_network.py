"""Correct: corridors exactly as the domain knowledge names them (West is 10th Ave E only, no
Broadway), with the instructed network step (networkx shortest paths to downtown). The half-mile
share of Capitol Hill is reported as a number only (~95.9 %); the service-area polygon is not kept.
"""
import geopandas as gpd, pandas as pd, matplotlib.pyplot as plt
nb = gpd.read_file("dataset/Neighborhoods.shp"); st = gpd.read_file("dataset/Streets.shp")
ch = nb[nb.L_HOOD == "CAPITOL HILL"].dissolve()
art = st[(st.SEGMENT_TY == 1) & (st.ARTERIAL_C == 1)]
near = art[art.intersects(ch.geometry.iloc[0].buffer(5))]
names = ["MADISON ST", "E MADISON ST", "10TH AVE E", "23RD AVE", "23RD AVE E", "24TH AVE E", "TURNER WAY E"]
routes = near[near.ORD_STNAME.isin(names)]
def _share(lines, hood):
    return lines.buffer(2640).union_all().intersection(hood).area / hood.area
coverage_share = _share(routes, ch.geometry.iloc[0])
print(f"share of Capitol Hill within half a mile of the proposed routes: {100 * coverage_share:.1f} %")
pd.DataFrame([{"metric": "coverage share", "value": coverage_share}]).to_csv("pred_results/coverage.csv", index=False)
routes.to_file("pred_results/proposed_routes.gpkg")
ax = ch.boundary.plot(color="k"); routes.plot(ax=ax, color="red")
ax.set_title("Proposed bike routes, Capitol Hill"); plt.savefig("pred_results/bike_routes.png")
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
