"""Correct: without a geocoder, the origin is the node nearest the centre of the network snapshot
(built around 1301 W Green St); osmnx speeds and travel times, drawn with ox.plot_graph.
"""
import osmnx as ox, networkx as nx, matplotlib.pyplot as plt
G = ox.load_graphml('dataset/uiuc_drive_network.graphml')
G = ox.add_edge_speeds(G); G = ox.add_edge_travel_times(G)
# no geocoder offline: the snapshot was built around 1301 W Green St, so use its centre
xs = [d['x'] for _, d in G.nodes(data=True)]; ys = [d['y'] for _, d in G.nodes(data=True)]
src = ox.nearest_nodes(G, X=(min(xs) + max(xs)) / 2, Y=(min(ys) + max(ys)) / 2)
distances = nx.single_source_dijkstra_path_length(G, src, weight='travel_time')
nx.set_node_attributes(G, distances, 'time2nhb')
for n in G.nodes:   # nodes unreachable on the directed graph get the maximum value
    G.nodes[n].setdefault('time2nhb', max(distances.values()))
nc = ox.plot.get_node_colors_by_attr(G, 'time2nhb', cmap='RdYlBu')
fig, ax = ox.plot_graph(G, node_size=40, node_color=nc, show=False, close=False)
fig.savefig('pred_results/travel_time.png')
