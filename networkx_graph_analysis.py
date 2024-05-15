import json
import networkx as nx
import matplotlib.pyplot as plt

# Load JSON data
with open('final_json_data.json') as f:
    data = json.load(f)

# Create a directed graph
G = nx.DiGraph()

# Add edges from the links section
for link in data['links']:
    source = link['source']
    target = link['target']
    label = link['label']
    G.add_edge(source, target, label=label)

# Compute degree for each node in the graph
degree_sequence = [(node, degree) for node, degree in G.degree()]

# Sort nodes by degree in descending order
degree_sequence.sort(key=lambda x: x[1], reverse=True)

# Print nodes and their degrees


print("Nodes sorted by degree (from highest to lowest):")
for node, degree in degree_sequence:
    print(f"Node: {node}, Degree: {degree}")


# Plot histogram of degree distribution for all nodes

all_degree_sequence = [d for n, d in degree_sequence]
plt.hist(all_degree_sequence, bins='auto', alpha=0.7, color='b', edgecolor='black')
plt.title("Degree Distribution for All Nodes")
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


betweenness_centrality = nx.betweenness_centrality(G)

top_20_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]

# Print top 20 nodes with highest betweenness centrality
print("Top 20 nodes with highest betweenness centrality:")
for node, centrality in top_20_nodes:
    print(f"Node: {node} : {centrality}")

"""
# Sort nodes by betweenness centrality in descending order
top_20_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]

# Extract top 20 node labels
top_20_node_labels = [node for node, _ in top_20_nodes]

# Create a subgraph containing only the top 20 nodes
G_top_20 = G.subgraph(top_20_node_labels)

# Draw the subgraph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_top_20, seed=42)  # positions for all nodes
nx.draw(G_top_20, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_color='black', edge_color='gray', width=0.5)
plt.title("Top 20 Nodes with Highest Betweenness Centrality")
plt.show()
"""

in_degree_centrality = nx.in_degree_centrality(G)

# Sort nodes by in-degree centrality in descending order
top_20_in_nodes = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]

# Print top 20 nodes with highest in-degree centrality
print("Top 20 nodes with highest in-degree centrality:")
for node, centrality in top_20_in_nodes:
    print(f"Node: {node} : {centrality}")

# Compute out-degree centrality for each node
out_degree_centrality = nx.out_degree_centrality(G)

# Sort nodes by out-degree centrality in descending order
top_20_out_nodes = sorted(out_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]

# Print top 20 nodes with highest out-degree centrality
print("\nTop 20 nodes with highest out-degree centrality:")
for node, centrality in top_20_out_nodes:
    print(f"Node: {node} : {centrality}")

# Extract top 20 node labels for in-degree and out-degree
top_20_in_node_labels = [node for node, _ in top_20_in_nodes]
top_20_out_node_labels = [node for node, _ in top_20_out_nodes]

# Create a subgraph containing only the top 20 nodes for in-degree and out-degree
G_top_20_in = G.subgraph(top_20_in_node_labels)
G_top_20_out = G.subgraph(top_20_out_node_labels)

# Draw the subgraphs
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
pos = nx.spring_layout(G_top_20_in, seed=42)
nx.draw(G_top_20_in, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_color='black', edge_color='gray', width=0.5)
plt.title("Top 20 Nodes with Highest In-Degree Centrality")

plt.subplot(1, 2, 2)
pos = nx.spring_layout(G_top_20_out, seed=42)
nx.draw(G_top_20_out, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_color='black', edge_color='gray', width=0.5)
plt.title("Top 20 Nodes with Highest Out-Degree Centrality")

plt.tight_layout()
plt.show()

eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)  # Increase max_iter
# Sort nodes by eigenvector centrality in descending order
top_20_eigenvector_nodes = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:20]

# Print top 20 nodes with highest eigenvector centrality
print("Top 20 nodes with highest eigenvector centrality:")
for node, centrality in top_20_eigenvector_nodes:
    print(f"Node: {node} : {centrality}")

# Extract top 20 node labels for eigenvector centrality
top_20_eigenvector_node_labels = [node for node, _ in top_20_eigenvector_nodes]

# Create a subgraph containing only the top 20 nodes for eigenvector centrality
G_top_20_eigenvector = G.subgraph(top_20_eigenvector_node_labels)

# Draw the subgraph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G_top_20_eigenvector, seed=42)  # positions for all nodes
nx.draw(G_top_20_eigenvector, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_color='black', edge_color='gray', width=0.5)
plt.title("Top 20 Nodes with Highest Eigenvector Centrality")
plt.show()