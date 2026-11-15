"""Reference solution for task 38: driving travel time from the Natural History Building (UIUC)
to every network node.

OSM drive network within 1 km of 1301 W Green St (frozen GraphML snapshot, no network in the
container) -> osmnx add_edge_speeds / add_edge_travel_times -> nearest node to the Natural
History Building (40.10944 N, 88.22625 W; coordinates used by the benchmark authors) ->
Dijkstra on edge travel_time (seconds, directed graph so one-way streets are respected) ->
node attribute 'time2nhb' -> nodes coloured by travel time.

Adapted from the reference script of GeoAnalystBench (Zhang et al. 2025, Apache-2.0).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

os.makedirs("pred_results", exist_ok=True)

G = ox.load_graphml("dataset/uiuc_drive_network.graphml")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

nhb_lat, nhb_lon = 40.109442588847976, -88.22625178710955
nhb_node = ox.nearest_nodes(G, X=nhb_lon, Y=nhb_lat)
times = nx.single_source_dijkstra_path_length(G, nhb_node, weight="travel_time")
nx.set_node_attributes(G, times, "time2nhb")

nodes, edges = ox.graph_to_gdfs(G)
nodes = nodes.reset_index()            # 'osmid' column
time2nhb = pd.Series(times, name="time2nhb")
print("origin node", nhb_node, "reachable", len(times), "of", len(G),
      "mean s", np.mean(list(times.values())), "max s", max(times.values()))

reach = nodes[nodes["time2nhb"].notna()]
fig, ax = plt.subplots(figsize=(10, 10))
edges.plot(ax=ax, color="#bbbbbb", linewidth=0.8, zorder=1)
sc = ax.scatter(reach["x"], reach["y"], c=reach["time2nhb"] / 60.0, cmap="RdYlBu_r", s=30, zorder=2,
                edgecolor="k", linewidth=0.2)
unr = nodes[nodes["time2nhb"].isna()]
if len(unr):
    ax.scatter(unr["x"], unr["y"], c="lightgrey", s=20, marker="x", zorder=2, label="unreachable")
ax.scatter([G.nodes[nhb_node]["x"]], [G.nodes[nhb_node]["y"]], marker="*", s=300, c="k", zorder=3,
           label="Natural History Building (origin)")
fig.colorbar(sc, ax=ax, shrink=0.7, label="driving time from Natural History Building (min)")
ax.legend(loc="lower left")
ax.set_title("Driving time to network nodes, UIUC campus")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.savefig("pred_results/travel_time.png", dpi=150, bbox_inches="tight")
