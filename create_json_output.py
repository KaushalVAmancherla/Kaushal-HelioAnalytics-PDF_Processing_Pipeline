import json

import argparse
import subprocess

def generate_json(subject_nodes_file, object_nodes_file, final_triples_file):
    nodes = []
    links = []

    # Read subject nodes
    with open(subject_nodes_file, 'r') as file:
        subject_nodes = [line.strip() for line in file]

    # Read object nodes
    with open(object_nodes_file, 'r') as file:
        object_nodes = [line.strip() for line in file]

    # Read triples
    with open(final_triples_file, 'r') as file:
        triples = [eval(line.strip()) for line in file]

    # Add subject nodes to nodes list
    for subject_node in subject_nodes:
        size = 0
        for triple in triples:
            if subject_node == triple[0]:
                size += 1
        placeholder_text = f"{subject_node} is linked to {size} object nodes."
        nodes.append({"id": subject_node, "group": 1, "size": size * 3, "placeholderText": placeholder_text})

    # Add object nodes to nodes list
    for object_node in object_nodes:
        size = 0
        for triple in triples:
            if object_node == triple[2]:
                size += 1
        placeholder_text = f"{object_node} is linked to {size} subject nodes."
        nodes.append({"id": object_node, "group": 2, "size": size * 3, "placeholderText": placeholder_text})

    # Add links
    for triple in triples:
        source = triple[0]
        target = triple[2]
        label = triple[1]
        links.append({"source": source, "target": target, "label": label})

    # Create JSON object
    json_output = {"nodes": nodes, "links": links}

    # Write JSON to file
    with open('final_json_data.json', 'w') as file:
        json.dump(json_output, file, indent=4)

    print("JSON output has been written to 'output.json' file.")

if __name__ == "__main__":
    print("REACHED JSON CURATION")

    generate_json('subject_nodes.txt', 'object_nodes.txt', 'final_triples.txt')
