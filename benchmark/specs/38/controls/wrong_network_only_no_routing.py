"""Wrong: colours the nodes by intersection degree; no shortest-path travel time from the origin is
computed.
"""
import osmnx as ox, matplotlib.pyplot as plt
G = ox.load_graphml('dataset/uiuc_drive_network.graphml')
G = ox.add_edge_speeds(G); G = ox.add_edge_travel_times(G)
# colours nodes by intersection degree; no shortest-path travel time from the origin
nc = ox.plot.get_node_colors_by_attr(G, 'street_count', cmap='RdYlBu')
fig, ax = ox.plot_graph(G, node_size=40, node_color=nc, show=False, close=False)
fig.savefig('pred_results/travel_time.png')
