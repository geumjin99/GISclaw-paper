"""Wrong: X and Y are swapped when the origin node is looked up, so the origin lies far from the NHB."""
import osmnx as ox, networkx as nx, matplotlib.pyplot as plt
G = ox.load_graphml('dataset/uiuc_drive_network.graphml')
G = ox.add_edge_speeds(G); G = ox.add_edge_travel_times(G)
src = ox.nearest_nodes(G, X=40.109442588847976, Y=-88.22625178710955)   # X/Y swapped
distances = nx.single_source_dijkstra_path_length(G, src, weight='travel_time')
nx.set_node_attributes(G, distances, 'time2nhb')
for n in G.nodes:   # nodes unreachable on the directed graph get the maximum value
    G.nodes[n].setdefault('time2nhb', max(distances.values()))
nc = ox.plot.get_node_colors_by_attr(G, 'time2nhb', cmap='RdYlBu')
fig, ax = ox.plot_graph(G, node_size=40, node_color=nc, show=False, close=False)
fig.savefig('pred_results/travel_time.png')
