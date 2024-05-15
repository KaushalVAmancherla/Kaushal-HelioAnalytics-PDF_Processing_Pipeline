import os
import argparse
import subprocess

#root_directory = "full_ADS_abstract_outputs"
#root_directory = "full_PDF_set_outputs"
#root_directory = "full_publication_outputs"
output_file = "first_triples.txt"

# Function to read strings from a text file
def read_strings_from_file(file_path):
    with open(file_path, 'r') as file:
        return {line.strip() for line in file}

# Function to iterate over subfolders and extract unique strings
def extract_unique_strings(root_dir):
    unique_strings = set()
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith('_triples.txt'):
                unique_strings.update(read_strings_from_file(file_path))
    return unique_strings

def create_dictionary_and_write(root_dir, output_file):
    unique_strings = extract_unique_strings(root_dir)
    with open(output_file, 'w') as file:
        for index, string in enumerate(unique_strings, start=1):
            file.write(f"{string}\n")

# Example usage:
#root_directory = "full_publication_outputs"
#output_file = "final_entities.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Processing Script")
    parser.add_argument("directory_name", help="Directory to read outputs from")
    args = parser.parse_args()

    if args.directory_name == "<directory_name>":
        print("Please provide the directory name to read outputs from.")
    else:
        directory_name = args.directory_name
        print("DIRECTORY PATH -> ", directory_name)

        create_dictionary_and_write(directory_name, output_file)

        resolve_acronyms_path = "./ads_acronym_resolver.py"
        subprocess.run(["python", resolve_acronyms_path])