import argparse
import spacy
import fitz
import nltk
import subprocess

from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize

extracted_text = []

nlp = spacy.load("en_core_web_sm")

def number_of_sentences(text):
    doc = nlp(text)
    sents = list(doc.sents)

    return len(sents)

def extract_text_from_pdf(pdf_path):
    text_paragraphs = dict()
    text_corpus = []

    with fitz.open(pdf_path) as doc:
        continual_string = ""

        start_consolidation = False
        end_consolidation = False

        for page in doc:
            key = str(page.number)                
            page_paragraphs = []

            blocks_text = [block[4] for block in page.get_text_blocks()]

            for text in blocks_text:
                text = text.strip()

                #print("Text -> ", text,"\n\n")
                
                if text and (text.strip()[-1] == '.' and text.strip()[0].istitle()) and number_of_sentences(text) >= 1:
                    page_paragraphs.append(text)
            
            text_paragraphs[key] = page_paragraphs

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
    args = parser.parse_args()

    if args.pdf_path == "<pdf_path>":
        print("Please provide the file path (PDF path).")
    else:
        try:
            file_name = args.pdf_path
            #print("FILE_NAME -> ", file_name)

            extracted_text = extract_text_from_pdf(file_name)

            create_outputs_path = "./create_outputs.py"
            subprocess.run(["python", create_outputs_path, *extracted_text])

            #print(len(extracted_text))

        except Exception as e:
            print(f"Error processing PDF: {e}")

#print(extracted_text)