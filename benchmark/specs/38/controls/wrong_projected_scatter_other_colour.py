"""Wrong: the travel times are computed correctly, but the UTM-projected node scatter is coloured by
node degree (street_count), not by travel time.
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
nx.set_node_attributes(G, tt, "travel_time")
trans = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
nodes = list(G.nodes())
px, py = trans.transform(np.array([G.nodes[n]["x"] for n in nodes]), np.array([G.nodes[n]["y"] for n in nodes]))
deg = np.array([float(G.nodes[n].get("street_count", 0)) for n in nodes])
fig, ax = plt.subplots(figsize=(9, 9))
sc = ax.scatter(px, py, c=deg, cmap="YlOrRd", s=40)
plt.colorbar(sc, ax=ax, label="Travel time")
plt.savefig("pred_results/travel_time.png", dpi=110, bbox_inches="tight")
