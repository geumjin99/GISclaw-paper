"""Wrong: travel time with one assumed city speed (40 km/h) for every edge; the speeds are not
derived from the data.
"""
import os, networkx as nx, osmnx as ox, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
G = ox.load_graphml("dataset/uiuc_drive_network.graphml")
for u, v, k, d in G.edges(keys=True, data=True):
    d["speed_kph"] = 40.0
    d["travel_time"] = d["length"] / (40.0 / 3.6)
src = ox.nearest_nodes(G, X=-88.22625, Y=40.10944)
tt = nx.single_source_dijkstra_path_length(G, src, weight="travel_time")
nx.set_node_attributes(G, tt, "travel_time")
xs = [G.nodes[n]["x"] for n in tt]; ys = [G.nodes[n]["y"] for n in tt]
plt.figure(figsize=(9, 9)); plt.scatter(xs, ys, c=list(tt.values()), cmap="viridis", s=25); plt.colorbar(label="s")
plt.savefig("pred_results/travel_time.png")
