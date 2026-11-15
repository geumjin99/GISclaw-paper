"""Correct: the origin is the node that a guessed NHB location snaps to (229 m from the NHB, within
the 400 m tolerance; the gold node is 116 m away); otherwise the reference solution.

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
nhb_node = 8881988386  # an NHB point guessed ~100-180 m off snaps to this S Wright St node, 229 m from the NHB
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
