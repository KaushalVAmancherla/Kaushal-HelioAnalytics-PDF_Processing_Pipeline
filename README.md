# Kaushal-HelioAnalytics-Summer_Internship
Extract entities and semantic triples on a Heliophysics based text (in PDF format) and graphically represent them

Environment Set-Up

1. Add your open ai api key in the .env file

2. Install the requirements.txt file to ensure all dependencies are covered for

Running the Script

To Process A Single PDF: 

1. Run the pipeline by the command: python process_pdf.py (insert the pdf file path here) (insert your chosen name for the file output folder) (insert your chosen name for the global output directory)

2. The model outputs will be stored in the generated folder/sub folder with two files : <file_output_name>_entities.txt and <file_output_name>_triples.txt

As is in the name, the <file_output_name>_entities.txt contains a list of all extracted semantic triple subject nodes, or entities, and <file_output_name>_triples.txt contains a list of all extracted semantic triples from the input pdf. 

E.g. python process_pdf.py /Users/kaushalamancherla/Downloads/9781119815617.pdf test_file test_output_directory 
will create a folder named test_output_directory. This folder will contain a subfolder in test_output_directory named test_file. The subfolder test_file will contain two files named: test_file_entities.txt and test_file_triples.txt 

To Process Multiple PDFs at once:

1. Run the pipeline by the command: python process_multiple_PDF.py (insert the directory of pdfs here) (insert your chosen name for the global output directory)

2. The model outputs will be stored in the generated folder/sub folder with two files : <file_name>_entities.txt and <file_name>_triples.txt

E.g. If I have a folder named papers that contain x number of PDFs each of which I want to run through the script, I run python process_multiple_PDF.py papers full_PDF_outputs

This creates a folder named full_PDF_outputs which contains x subfolders named after the processed file. Each subfolder contains the two files <file_name>_entities.txt and <file_name>_triples.txt

To Generate The Graph Outputs:

1. Run python curate_final_lists.py (path to global directory containing the subfolders storing model outputs). This creates several files:

final_triples.txt -> The final list of all semantic triples consolidated over the inputs 
object_nodes.txt -> The final list of all object nodes consolidated over the inputs 
subject_nodes.txt -> The final list of all subject nodes consolidated over the inputs 
final_json_data.json -> The json data used for graph curation

replacement_stats.txt -> Text file containing the data for synonym resolution. Formatted in the following manner: 
(synonym used for replacement) -> (number of instances that the synonym replaced a word), [list of each word instance that the synonym replaced]

acronyms_replacement.txt -> Text file containing the data for acronym resolution. Formatted in the following manner:
(acronym) -> (number of times this acronym got expanded to its full, written form)

2. Run python3 -m http.server
3. Open http://localhost:8000/d3_graph.html in any broswer to see the network graph
