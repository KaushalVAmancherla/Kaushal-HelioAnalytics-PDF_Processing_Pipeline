from create_text_outputs import extract_words

import argparse
import networkx as nx
import matplotlib.pyplot as plt
import os
import json

graph = nx.DiGraph()

nodes = set()
node_labels = dict()
node_sizes = dict()
edge_labels = set()

edge_label_subject = set()
edge_label_object = set()

filtered_entities_dict = dict()

subfolders = ["betweeness", "degree", "indegree","closeness","pagerank","voterank"] 

subfiles = ["betweeness_nodes.txt","degree_nodes.txt","indegree_nodes.txt","closeness_nodes.txt",
                        "pagerank_nodes.txt","voterank_nodes.txt"]

subfiles_graphs = ["betweeness_centrality.png","degree_centrality.png","indegree_centrality.png",
            "closeness_centrality.png","pagerank_centrality.png","voterank_centrality.png"]

main_folder = "graphical_outputs"  
centrality_folder = "centrality_outputs"  
kg_folder = "kg_outputs"  

graphical_outputs_directory = os.path.join(os.getcwd(),main_folder)
centrality_outputs_folder = os.path.join(graphical_outputs_directory, centrality_folder)
kg_outputs_folder = os.path.join(graphical_outputs_directory,kg_folder)

def create_output_folders():
    os.makedirs(centrality_outputs_folder, exist_ok=True)
    os.makedirs(kg_outputs_folder, exist_ok=True)

    for idx,subfolder in enumerate(subfolders):
        subfolder_path = os.path.join(centrality_outputs_folder, subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

        file_name = subfiles[idx]
        file_path = os.path.join(subfolder_path, file_name)

        with open(file_path,'w') as file:
            file.write("")

    print("Folders/files created successfully.")

def consolidate_graph_data(filtered_entities_dict):
    for val in filtered_entities_dict:
        score = filtered_entities_dict[val][0]

        for triple in filtered_entities_dict[val][1]:
            ents_list = extract_words(triple)

            nodes.add(ents_list[0])
            nodes.add(ents_list[2])

            node_labels[ents_list[0]] = ents_list[0]
            node_labels[ents_list[2]] = ents_list[2]

            node_sizes[ents_list[0]] = score * 10
            node_sizes[ents_list[2]] = score * 10

            edge_labels.add((ents_list[0],ents_list[2]))

            edge_label_subject.add(ents_list[0])
            edge_label_object.add(ents_list[2])

            graph.add_edge(ents_list[0], ents_list[2], label=ents_list[1])

    graph.add_nodes_from(nodes)
    node_sizes_ordered = [node_sizes[node] for node in graph.nodes()]

    pos = nx.spring_layout(graph)     
    
    
    graph_degree_freq(graph)
    draw_original_graph(graph,pos,node_sizes_ordered,node_labels)    
    
    #Centrality Measures
    
    betweeness_cen = nx.betweenness_centrality(graph)
    degree_cen = nx.degree_centrality(graph)
    in_deg_cen = nx.in_degree_centrality(graph)
    closeness_cen = nx.closeness_centrality(graph)
    pr_cen = nx.pagerank(graph)
    vote_rank = nx.voterank(graph)

    draw(betweeness_cen,"Betweeness Centrality",pos,node_sizes_ordered,0,0)
    draw(degree_cen,"Degree Centrality",pos,node_sizes_ordered,1,1)
    draw(in_deg_cen,"In-Degree Centrality",pos,node_sizes_ordered,2,2)
    draw(closeness_cen,"Closeness Centrality",pos,node_sizes_ordered,3,3)
    draw(pr_cen,"Page Rank Centrality",pos,node_sizes_ordered,4,4)    
    draw_vote_rank(vote_rank,"Vote Rank Centrality",nodes,pos,node_sizes_ordered,5,5)
    
    
def write_dict_to_file(file_path,dictionary):
    with open(file_path, 'w') as file:
        for key,value in dictionary.items():
            file.write(f"{key}: {value}\n")

def write_nodes_to_file(file_path,nodes_list):
    with open(file_path, 'w') as file:
        for node in nodes_list:
            file.write(f"{node}\n")

def draw_original_graph(graph,pos,node_sizes_ordered,node_labels):
    nx.draw_networkx(graph,pos,font_size = 8,node_size = node_sizes_ordered, labels = node_labels, with_labels=True)  
    nx.draw_networkx_edge_labels(graph,pos,font_size = 5,edge_labels=nx.get_edge_attributes(graph,'label'))

    plt.title("Knowledge Graph",fontsize = 13)

    kg_file_path = os.path.join(kg_outputs_folder,'knowledge_graph.png')
    plt.savefig(kg_file_path)

    plt.show()

def draw(measures,name,pos,node_sizes_ordered,folder_idx,graph_idx):
    graph = nx.DiGraph()

    graph.add_nodes_from(measures.keys())
    graph.add_edges_from(edge_labels)

    top_nodes = []

    measures_modified = {node:centrality for node,centrality in measures.items() if node in edge_label_subject}
    measures_modified = dict(sorted(measures_modified.items(), key=lambda item: item[1], reverse=True))

    top_scores = sorted(set(measures_modified.values()), reverse=True)[:3]
    top_nodes = [key for key, value in measures_modified.items() if value in top_scores]
    
    top_dict = {key:value for key,value in measures_modified.items() if value in top_scores}
    top_dict = dict(sorted(top_dict.items(), key=lambda item: item[1], reverse=True))

    dict_file_path = os.path.join(centrality_outputs_folder + '/' + subfolders[folder_idx],subfiles[graph_idx])
    write_dict_to_file(dict_file_path,top_dict)

    labels = {node: node for node in top_nodes}

    nodes = nx.draw_networkx_nodes(graph, pos, node_color=list(measures.values()), cmap=plt.cm.plasma,node_size = node_sizes_ordered)

    nx.draw_networkx_labels(graph, pos, font_size = 8, labels=labels)
    nx.draw_networkx_edges(graph, pos)

    plt.colorbar(nodes)  
    plt.axis('off')  
    plt.title(name,fontsize = 13)

    file_path = os.path.join(centrality_outputs_folder + '/' + subfolders[folder_idx],subfiles_graphs[graph_idx])

    plt.savefig(file_path)
    plt.show()
    
def draw_vote_rank(vote_rank,name,nodes,pos,node_sizes_ordered,folder_idx,graph_idx):
    graph = nx.DiGraph()

    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_labels)

    colors = [node_sizes[node] for node in nodes]
    nodes_return = nx.draw_networkx_nodes(graph, pos, node_color=colors, cmap=plt.cm.plasma,node_size = node_sizes_ordered)

    labels = {node: node for node in vote_rank if node in edge_label_subject}
    labels_node = [node for node in labels]

    node_list_file_path = os.path.join(centrality_outputs_folder + '/' + subfolders[folder_idx],subfiles[graph_idx])
    write_nodes_to_file(node_list_file_path,labels_node)

    nx.draw_networkx_labels(graph, pos, font_size = 8, labels=labels)
    nx.draw_networkx_edges(graph, pos)

    plt.colorbar(nodes_return)  # Add a colorbar
    plt.axis('off')  # Disable axis display
    plt.title(name,fontsize = 13)

    file_path = os.path.join(centrality_outputs_folder + '/' + subfolders[folder_idx],subfiles_graphs[graph_idx])
    plt.savefig(file_path)

    plt.show()

def graph_degree_freq(graph):
    degree_freq = nx.degree_histogram(graph)
    degrees = range(len(degree_freq))

    #filtered_degree_freq = [count for degree, count in enumerate(degree_freq) if degree > 1]
    #filtered_degrees = range(2, len(filtered_degree_freq) + 2)  # Start from degree 2

    #print("Degree distribution data -> ", filtered_degree_freq)

    # Plot the histogram
    plt.bar(degree_freq, degrees)

    # Set labels and title
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')

    hist_file_path = os.path.join(kg_outputs_folder,'degree_histogram.png')
    plt.savefig(hist_file_path)

    # Show the plot
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Entities/Triples")
    parser.add_argument("dict",type=str,nargs="+", help="Dictionary mapping each entity to their semantic triples")
    args = parser.parse_args()
    
    create_output_folders()

    filtered_entities_dict = eval(args.dict[0])
    consolidate_graph_data(filtered_entities_dict)

    """
    print(type(filtered_entities_dict))

    for val in my_dict:
        print(val,"->",my_dict[val],"\n\n\n")
    
    print(nodes)
    """