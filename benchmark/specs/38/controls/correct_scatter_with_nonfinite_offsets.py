"""Correct: osmnx speeds and travel times; the node scatter includes every graph node, unreachable
ones with NaN coordinates and a NaN value (non-finite offsets).
"""
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

G = ox.load_graphml("dataset/uiuc_drive_network.graphml")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
origin = ox.distance.nearest_nodes(G, -88.226252, 40.109443)
times = nx.single_source_dijkstra_path_length(G, origin, weight="travel_time")
nx.set_node_attributes(G, times, "travel_time")
nodes = list(G.nodes)
xs = np.array([G.nodes[n]["x"] if n in times else np.nan for n in nodes])
ys = np.array([G.nodes[n]["y"] if n in times else np.nan for n in nodes])
vals = np.array([times.get(n, np.nan) for n in nodes])
fig, ax = plt.subplots(figsize=(9, 9))
for u, v in G.edges():
    ax.plot([G.nodes[u]["x"], G.nodes[v]["x"]], [G.nodes[u]["y"], G.nodes[v]["y"]], color="#bbb", lw=0.6, zorder=1)
sc = ax.scatter(xs, ys, c=vals, cmap="viridis", s=20, zorder=5)
fig.colorbar(sc, ax=ax, label="travel time (s)")
fig.savefig("pred_results/travel_time.png", dpi=120)
