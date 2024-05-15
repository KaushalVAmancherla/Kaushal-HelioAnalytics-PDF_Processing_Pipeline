import argparse
import spacy
import fitz
import nltk
import subprocess
import os 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
from stanfordcorenlp import StanfordCoreNLP

extracted_text = []

gpt4_outputs_path = ""
gpt4_entities_path = ""

gpt4_outputs_list = []
gpt4_entities_list = []

nlp = spacy.load("en_core_web_md")

#output_directory = "full_ADS_abstract_outputs"
#output_directory = "full_PDF_set_outputs"
#pub_output_directory = "full_publication_outputs"

output_directory = ""

def create_or_clear_files(pdf_path, dir_file_name,dir_name):
    directory = os.getcwd()

    #define the outputs
    output_directory = os.path.join(directory,dir_name)
    print("OUTPUT DIRECTORY -> "), output_directory

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Define the subfolder path based on dir_file_name
    subfolder_path = os.path.join(output_directory, dir_file_name)

    # Create the subfolder if it doesn't exist
    if not os.path.exists(subfolder_path):
        #print("REACHED_1")
        os.makedirs(subfolder_path)

    # Define the paths for the files within the subfolder
    gpt4_entities_path = os.path.join(subfolder_path, f"{dir_file_name}_entities.txt")
    gpt4_outputs_path = os.path.join(subfolder_path, f"{dir_file_name}_triples.txt")

    print("ENTITIES FILE -> ", gpt4_entities_path)
    print("OUTPUTS FILE -> ", gpt4_outputs_path)

    gpt4_outputs_list.append(gpt4_outputs_path)
    gpt4_entities_list.append(gpt4_entities_path)

    # Create or clear the entities file
    with open(gpt4_entities_path, 'w') as f:
        f.write("")

    # Create or clear the outputs file
    with open(gpt4_outputs_path, 'w') as f:
        f.write("")

    
def number_of_sentences(text):
    doc = nlp(text)
    sents = list(doc.sents)

    return len(sents)

def extract_text_from_pdf(pdf_path):
    text_paragraphs = dict()
    text_corpus = []

    with fitz.open(pdf_path) as doc:
        continual_string = ""
        stop_processing = False  # Flag to stop processing further text blocks

        for page in doc:
            key = str(page.number)                
            page_paragraphs = []

            blocks_text = [block[4] for block in page.get_text_blocks()]

            #print("PAGE NUMBER -> ", key,"\n\n")

            for text in blocks_text:
                text = text.strip()

                if any(title_word in text.lower() for title_word in ["reference", "appendix","acknowledgments","acknowledgements"]):
                    stop_processing = True
                    break

                if text and (text.strip()[-1] == '.' or text.strip()[0].istitle()) and number_of_sentences(text) > 1:
                    #print("APPENDED TEXT -> ", text,"\n\n\n")
                    page_paragraphs.append(text)


            #print("PAGE PARAGRAPH -> ", page_paragraphs,"\n\n\n")
            text_paragraphs[key] = page_paragraphs

            if stop_processing:
                break

    continual_string = ""

    for key in text_paragraphs:
        for value in text_paragraphs[key]:
            sentences = sent_tokenize(value)

            continual_string += value

            if sentences[-1].endswith('.'):
                text_corpus.append(continual_string)
                continual_string = ""

    extracted_text_str = create_string_from_list(text_corpus)
    #print(extracted_text_str)

    text_corpus_final = split_text(extracted_text_str)

    if not text_corpus_final:
        raise Exception("List is empty, cannot proceed to the next script")

    return text_corpus_final



def create_string_from_list(text_corpus):
    extracted_text_str = ""

    for paragraph in text_corpus:
        extracted_text_str += (paragraph + "\n\n")

    return extracted_text_str

def split_text(text):
    text_corpus_final = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 750,
        chunk_overlap  = 80,
        length_function = len,
        add_start_index = False,
    )

    texts = text_splitter.create_documents([text])
    
    for sent in texts:
        text_corpus_final.append(sent.page_content)
        #print(sent.page_content,"\n\n")

    return text_corpus_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF Processing Script")
    parser.add_argument("pdf_path", help="Directory path to the pdf file")
    parser.add_argument("pdf_name", help="Name of your pdf to put in entry")
    parser.add_argument("output_directory", help="Name of the output_directory to store the outputs")
    args = parser.parse_args()

    if args.pdf_path == "<pdf_path>":
        print("Please provide the file path (PDF path).")
    elif args.pdf_name == "<pdf_name>":
        print("Please provide the file name you want entered (PDF name).")
    elif args.output_directory == "<output_directory>":
        print("Please provide the output directory name you want entered.")
    else:
        try:
            file_path = args.pdf_path
            print("FILE_PATH -> ", file_path)

            file_name = args.pdf_name
            print("FILE_NAME -> ", file_name)

            output_directory = args.output_directory
            print("OUTPUT DIRECTORY -> ", output_directory)

            create_or_clear_files(args.pdf_path, args.pdf_name,output_directory)

            extracted_text = extract_text_from_pdf(file_path)
            
            print("EXTRACTED TEXT", extracted_text)
            #print("GOING TO NEXT VIEW")

            create_outputs_path = "./create_text_outputs.py"
            subprocess.run(["python", create_outputs_path, *extracted_text,*gpt4_entities_list,*gpt4_outputs_list])
            
        except Exception as e:
            print(f"Error processing PDF: {e}")

#print(extracted_text)