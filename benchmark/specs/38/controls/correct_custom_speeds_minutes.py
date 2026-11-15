"""Correct: own speed model (posted maxspeed, else US urban defaults per road class), Dijkstra in
networkx, travel times in minutes.
"""
import re
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt

DEFAULT_MPH = {"primary": 30, "secondary": 30, "tertiary": 25, "residential": 25, "unclassified": 25}

def mph(d):
    m = re.search(r"\d+", str(d.get("maxspeed") or ""))
    if m:
        return float(m.group())
    hw = d.get("highway"); hw = hw[0] if isinstance(hw, list) else str(hw)
    return DEFAULT_MPH.get(hw.replace("_link", ""), 25)

def analyse():
    G = ox.load_graphml("dataset/uiuc_drive_network.graphml")
    for u, v, k, d in G.edges(keys=True, data=True):
        d["minutes"] = d["length"] / (mph(d) * 1609.344 / 60.0)
    origin = ox.distance.nearest_nodes(G, -88.226252, 40.109443)
    mins = nx.single_source_dijkstra_path_length(G, origin, weight="minutes")
    nx.set_node_attributes(G, mins, "drive_min")
    fig, ax = ox.plot_graph(G, node_color=[plt.cm.magma_r(mins.get(n, 0) / max(mins.values())) for n in G.nodes],
                            node_size=25, show=False, close=False, bgcolor="white", edge_color="#999")
    ax.scatter([G.nodes[n]["x"] for n in mins], [G.nodes[n]["y"] for n in mins], c=list(mins.values()),
               cmap="magma_r", s=25, zorder=5)
    fig.savefig("pred_results/travel_time.png", dpi=120)

analyse()
