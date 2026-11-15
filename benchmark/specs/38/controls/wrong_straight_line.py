"""Wrong: 'travel time' = straight-line distance from the NHB / typical speed (no network routing)."""
import os, numpy as np, osmnx as ox, networkx as nx, matplotlib.pyplot as plt
os.makedirs("pred_results", exist_ok=True)
G = ox.load_graphml("dataset/uiuc_drive_network.graphml")
G = ox.add_edge_speeds(G); G = ox.add_edge_travel_times(G)
lat0, lon0 = 40.10944, -88.22625
src = ox.nearest_nodes(G, X=lon0, Y=lat0)
mean_speed = np.mean([d["speed_kph"] for _, _, d in G.edges(data=True)]) / 3.6
times = {}
for n, d in G.nodes(data=True):
    dist = ox.distance.great_circle(G.nodes[src]["y"], G.nodes[src]["x"], d["y"], d["x"])
    times[n] = dist / mean_speed
nx.set_node_attributes(G, times, "travel_time")
nodes = ox.graph_to_gdfs(G, edges=False).reset_index()
ax = nodes.plot(column="travel_time", cmap="plasma", legend=True, markersize=20, figsize=(9, 9))
plt.savefig("pred_results/travel_time.png")
