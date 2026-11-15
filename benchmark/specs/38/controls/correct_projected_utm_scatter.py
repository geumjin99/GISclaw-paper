"""Correct: osmnx speeds and travel times, Dijkstra from the node nearest the NHB, node travel times
stored on the graph; the map is drawn after projecting node and edge coordinates to UTM 16N with
pyproj (scatter coloured by travel time).
"""
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
from pyproj import Transformer

G = ox.load_graphml("dataset/uiuc_drive_network.graphml")
G = ox.add_edge_travel_times(ox.add_edge_speeds(G))
source = ox.distance.nearest_nodes(G, -88.226252, 40.109443)
tt = nx.single_source_dijkstra_path_length(G, source, weight="travel_time")
for n, t in tt.items():
    G.nodes[n]["travel_time"] = t
trans = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
nodes = list(G.nodes())
px, py = trans.transform(np.array([G.nodes[n]["x"] for n in nodes]), np.array([G.nodes[n]["y"] for n in nodes]))
TT = np.array([G.nodes[n].get("travel_time", np.nan) for n in nodes], float)
fig, ax = plt.subplots(figsize=(9, 9))
for u, v, d in G.edges(data=True):
    ex, ey = trans.transform([G.nodes[u]["x"], G.nodes[v]["x"]], [G.nodes[u]["y"], G.nodes[v]["y"]])
    ax.plot(ex, ey, color="0.8", lw=1, zorder=1)
m = ~np.isnan(TT)
sc = ax.scatter(px[m], py[m], c=TT[m], cmap="YlOrRd", s=40, zorder=3, edgecolors="k", linewidths=0.3)
plt.colorbar(sc, ax=ax, label="Driving travel time (s)")
ax.set_aspect("equal"); ax.set_xlabel("Easting (m, UTM 16N)")
plt.savefig("pred_results/travel_time.png", dpi=110, bbox_inches="tight")
