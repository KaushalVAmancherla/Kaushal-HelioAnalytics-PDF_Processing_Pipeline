import argparse
import subprocess

triples_file = 'final_triples.txt'

output_file_obj = 'object_nodes.txt'
output_file_subj = 'subject_nodes.txt'

def extract_object_nodes(triples_file):
    object_nodes = set()
    with open(triples_file, 'r') as file:
        for line in file:
            triple = eval(line.strip())  
            object_nodes.add(triple[2])  
    return object_nodes

def write_node(nodes, output_file):
    with open(output_file, 'w') as file:
        for node in nodes:
            file.write(node + '\n')

def extract_subject_nodes(triples_file):
    object_nodes = set()
    with open(triples_file, 'r') as file:
        for line in file:
            triple = eval(line.strip())  
            object_nodes.add(triple[0])  
    return object_nodes

if __name__ == "__main__":
    print("REACHED EXTRACT NODE")

    object_nodes = extract_object_nodes(triples_file)
    subject_nodes = extract_subject_nodes(triples_file)

    write_node(object_nodes, output_file_obj)
    write_node(subject_nodes, output_file_subj)

    curate_json = "./create_json_output.py"
    subprocess.run(["python", curate_json])