"""Wrong: Dijkstra on edge length (metres) instead of travel time."""
import osmnx as ox, networkx as nx, matplotlib.pyplot as plt
G = ox.load_graphml('dataset/uiuc_drive_network.graphml')
src = ox.nearest_nodes(G, X=-88.22625, Y=40.10944)
distances = nx.single_source_dijkstra_path_length(G, src, weight='length')   # metres, not time
nx.set_node_attributes(G, distances, 'time2nhb')
for n in G.nodes:   # nodes unreachable on the directed graph get the maximum value
    G.nodes[n].setdefault('time2nhb', max(distances.values()))
nc = ox.plot.get_node_colors_by_attr(G, 'time2nhb', cmap='RdYlBu')
fig, ax = ox.plot_graph(G, node_size=40, node_color=nc, show=False, close=False)
fig.savefig('pred_results/travel_time.png')
