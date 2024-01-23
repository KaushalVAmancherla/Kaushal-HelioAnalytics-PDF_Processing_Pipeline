import os
import re
import fitz
import csv
import json

import openai
import spacy
import nltk
import tiktoken

import argparse
import subprocess

from PIL import Image
from nltk.tokenize import sent_tokenize
from nltk.wsd import lesk
from nltk.corpus import wordnet
#from pywsd.lesk import adapted_lesk

from openie import StanfordOpenIE
from sklearn.feature_extraction.text import TfidfVectorizer

from spacy.tokens import Token
from spacy.lang.en.stop_words import STOP_WORDS

from collections import defaultdict
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

#nltk.download('wordnet')

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

helio_acronyms_list_path = './Helio-KNOW/ADS_enrichment/data/solar_physics_acronyms.csv'
yamz_file_path_ont = "./yamz/uploads/helio"

gdc_report_path = './GDC_STDT_Report_FINAL.pdf'


gpt4_entities_path = "gpt4_entities.txt"
gpt4_outputs_path = "gpt4_outputs.txt"
gpt4_outputs_processed_path = "gpt4_outputs_processed.txt"

gpt4_graph_entities_path = "gpt4_graph_entities.txt"
gpt4_graph_outputs_path = "gpt4_graph_outputs.txt"

"""
gpt4_entities_path = "gpt4_entities_GDC.txt"
gpt4_outputs_path = "gpt4_outputs_GDC.txt"

gpt4_graph_entities_path = "gpt4_graph_entities_GDC.txt"
gpt4_graph_outputs_path = "gpt4_graph_outputs_GDC.txt"
"""

stop_words_getter = lambda token: token.is_stop or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter)

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("entityLinker", last=True)

system_msg = "You are a language model that performs relation extraction on a heliophysics publication to extract semantic triples."

query = "In the following paragraph, extract named heliophysics entities and their relationships as triples like so: ('subject', 'predicate', 'object'). Ensure the following at all times: necessary/relevant/explicit context is added/kept and the triples contain clear definitions and zero ambiguity, only composite/multi-word object triples (not basic/simple triples) are considered, only consider predicates with verbs that aren’t gerunds/-ing verbs so that the triples are complete/informative, entities are kept in their complete, expanded form without any abbreviation, and the context of comparison regarding the entity is kept when using comparatives."
example = "Given the sentence 'mesoscale processes substantially contribute to processes in surrounding regions', extract ('mesoscale processes’, ‘substantially contribute to’, ‘processes in surrounding region’). Given ‘Arcs are associated with enhanced precipitation, FAC, and electric field’, extract (‘Arcs','are associated with’,’enhanced precipitation, FAC, and electric field’)"

messages_query=[
    {"role": "system", "content": system_msg},
    {"role": "user", "content": query},
    {"role": "assistant", "content": example}
]

model_gpt4 = "gpt-4"

properties = {
    'openie.affinity_probability_cap': 0.9,
}

omit_set = {'PRON'}
predicate_set = {'AUX','VERB'}

gpt4_final_list_entity = []
gpt4_similarity_dict = defaultdict(set)
gpt4_prefinal_set = set()
gpt4_final_set = set()

glossary_set = set()
glossary_dict_processed = dict()

acronyms_set = set()
yamz_set = set()

helio_acronyms_set = set()
helio_acronyms_set_expanded = set()
helio_acronyms_set_dict = dict()

gdc_acronyms = []
gdc_acronyms_exapnded = []
gdc_acronym_dict = dict()

num_ents = 0
num_triples = 0

page_number = 107

lemmatized_verb_dict = dict()
entity_linkage_dict = dict()
entity_disambiguation_dict = dict()

"""
def consolidate_acronyms(pdf_path,page_number):
    with fitz.open(pdf_path) as doc:
        page = doc[page_number]
        #text = page.get_text()
        
        blocks_text = [block[4] for block in page.get_text_blocks()]

        for text in blocks_text:
            text_arr = text.strip().split("\n")

            if len(text_arr) > 1:
                key = text_arr[0]
                value = ''.join(text_arr[1:])

                gdc_acronyms.append(key)
                gdc_acronyms_exapnded.append(value)
                gdc_acronym_dict[key] = value
"""

def acronym_set_curation():
    with open(helio_acronyms_list_path) as file_obj:
        reader_obj = csv.reader(file_obj)

        for row_arr in reader_obj:
            helio_acronyms_set.add(row_arr[0])
            helio_acronyms_set_expanded.add(row_arr[1])
            helio_acronyms_set_dict[row_arr[0]] = row_arr[1]
    
def curate_yamz_set():
    for filename in os.listdir(yamz_file_path_ont):
        file_path = os.path.join(yamz_file_path_ont, filename)
        
        with open(file_path, 'r') as file:
            data = json.load(file)

            if "Semantic Web for Earth and Environment Technology Ontology" not in data['Source Name']:
                for val in data['Terms']:
                    yamz_set.add(val['Term'])

def consolidate_glossary_set():
    global glossary_set
    global glossary_set_processed
    global acronyms_set

    glossary_set = yamz_set | helio_acronyms_set | helio_acronyms_set_expanded #| set(gdc_acronyms) | set(gdc_acronyms_exapnded)
    acronyms_set = helio_acronyms_set | helio_acronyms_set_expanded #| set(gdc_acronyms) | set(gdc_acronyms_exapnded)

    for entity in glossary_set:
        glossary_dict_processed[word_process(entity)] = entity

def contains_pronouns_stopwords(extracted_text):
    doc = nlp(extracted_text)    

    for token in doc:
        #print("token -> ", token," ",token.pos_)
        
        if token.pos_ in omit_set or token._.is_stop:
            #print("OMIT THIS SUBJECT\n")
            return True

    return False

def is_valid_predicate(extracted_text):
    doc = nlp(extracted_text)
    
    #lemma_list = [token.lemma_ for token in doc]
    #doc = nlp(' '.join(lemma_list))    

    for token in doc:
        if token.pos_ in predicate_set: return True
        #print("token -> ", token,token.pos_,token.pos_ in predicate_set)

    return False

def contains_modal_verb(extracted_text):
    doc = nlp(extracted_text)

    for token in doc:
        if token.dep_.lower() == 'aux' and token.tag_.lower() == 'md': 
            #print("token -> ", token,"\n")
            return True
            
    return False

def process_triple(subject,relation,obj):
    return [subject.lower().replace("-"," ").replace("\n"," "),relation.lower().replace("-"," ").replace("\n"," "),obj.lower().replace("-"," ").replace("\n"," ")]

def word_process(text):
    text = text.lower().replace('-',' ')
    doc = nlp(text)
    
    lemma_list = []
    for token in doc:
        #print(token.text, token.lemma_, ps.stem(token.text))
        lemma_list.append(token.lemma_)

    text = ' '.join(lemma_list)
    return text

def compute_related(outer_val,inner_val):
    similarity_ratio_subject = fuzz.ratio(outer_val[0], inner_val[0])
    similarity_ratio_predicate = fuzz.ratio(outer_val[1], inner_val[1])
    similarity_ratio_object = fuzz.ratio(outer_val[2], inner_val[2])

    #print(similarity_ratio_subject,similarity_ratio_predicate,similarity_ratio_object)

    if similarity_ratio_subject > 70 and similarity_ratio_predicate > 70 and similarity_ratio_object > 70: 
        average_score = (similarity_ratio_subject + similarity_ratio_predicate + similarity_ratio_object)/3

        #print(outer_relation_processed,inner_relation_processed,"similarity_score_avg -> ", average_score)
        return average_score

    #return False

    return -1

def compute_relation_similarity(relation_list,relation_dict,similarity_dict):    
    for i in range(len(relation_list) - 1):
        relation = relation_list[i]
        relation_len = len(relation[0]) + len(relation[1]) + len(relation[2])

        #print(relation,"<<<|OUTER RELATION|>>>")

        similarity_dict[(relation,relation_len)] = list()

        for j in range(i+1,len(relation_list)):
            inner_relation = relation_list[j]
            inner_relation_len = len(inner_relation[0]) + len(inner_relation[1]) + len(inner_relation[2])

            similarity_score = compute_related(relation_dict[relation],relation_dict[inner_relation])
            
            if similarity_score != -1:
                #print("MATCH -> ", relation_dict[relation]," <-> ", relation_dict[inner_relation],similarity_score)
                similarity_dict[(relation,relation_len)].append((inner_relation,inner_relation_len))

        #print("\n")
        #print(relation,relation_dict[relation])

def extract_words(text):
    first_quote_index = text.find("'")
    second_quote_index = text.find("'", first_quote_index + 1)
    third_quote_index = text.find("'", second_quote_index + 1)
    fourth_quote_index = text.find("'", third_quote_index + 1)
    fifth_quote_index = text.find("'", fourth_quote_index + 1)
    sixth_quote_index = text.find("'", fifth_quote_index + 1)

    # Extract the text between the single quotes
    subj = text[text.find("'") + 1:text.find("'", first_quote_index + 1)]
    pred = text[text.find("'", second_quote_index + 1) + 1:text.find("'", third_quote_index + 1)]
    obj = text[text.find("'", fourth_quote_index + 1) + 1:text.find("'", fifth_quote_index + 1)]

    return [subj,pred,obj]

def num_tokens_from_messages(messages, model):
    encoding = tiktoken.encoding_for_model(model)

    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

def filter_list(related_dict,relation_list,final_set):
    copy_relation_list = list(relation_list)

    #print("len -> ", len(copy_relation_list))

    for key in related_dict:
        value_list = []
        for item in related_dict[key]: value_list.append(item[0])

        if key[0] in copy_relation_list and (all(item in copy_relation_list for item in value_list) or not related_dict[key]):
            #print("key -> ", key[0], "| value_list -> ", value_list)

            inner_list = []
            inner_list.append(key)

            for value in related_dict[key]:
                inner_list.append(value)

            inner_list = sorted(inner_list, key=lambda item: item[1], reverse=True)
            final_set.add(inner_list[0][0])

            #print(inner_list[0][0])

            for delete_element in inner_list:
                if delete_element[0] in copy_relation_list:
                    copy_relation_list.remove(delete_element[0])

    #print("total list before -> ", len(relation_list))
    #print("total list after -> ", len(final_set))
    
def process_gpt(openai_outputs,gpt_entities_path,gpt_outputs_path):
    for str_tuple in openai_outputs: 
        start_index = str_tuple.find('(')
        end_index = str_tuple.rfind(')')
        
        substring = str_tuple[start_index:end_index+1].strip('()')

        quote_count = substring.count("'")

        if quote_count == 6:
            #print("tuple -> ", substring)

            subject,predicate,obj = extract_words(substring)
            subject_copy,predicate_copy,obj_copy = process_triple(subject,predicate,obj)

            if not contains_modal_verb(predicate_copy) and not contains_pronouns_stopwords(subject_copy) and is_valid_predicate(predicate_copy):
                #print(subject,"|",predicate,"|",obj)

                with open(gpt_entities_path,'a') as file:
                    file.write(subject + "\n")

                with open(gpt_outputs_path,'a') as file:
                    file.write(str((subject,predicate,obj)) + "\n")       

def extract_gpt(paragraph_num,text,model_spec,gpt_entities_path,gpt_outputs_path):
    #print("PARAGRAPH -> ",paragraph_num, "| model -> ",model_spec)
    #print("text -> ",text)

    messages_query[1]["content"] = query + " " + text
    #print(num_tokens_from_messages(messages_query,model_spec))

    
    chat_completion = openai.ChatCompletion.create(
        model=model_spec, messages=messages_query, temperature = 0
    )

    output = chat_completion.choices[0].message.content    

    if "\n" in output:
        #print("PARAGRAPH -> ",paragraph_num, "| model -> ",model_spec,"\n", "| TOKENS -> ", chat_completion.usage.total_tokens)
        
        output = output.split("\n")
        #print(output,"\n\n\n")

        process_gpt(output,gpt_entities_path,gpt_outputs_path)
    
def process_extracted_text_gpt(extracted_text,model_spec,gpt_entities_path,gpt_outputs_path):
    for line,paragraph in enumerate(extracted_text):
        #print("paragraph -> ", paragraph,"\n\n")
        extract_gpt(line,paragraph,model_spec,gpt_entities_path,gpt_outputs_path)

def write_processed_relations(file_path,processed_file_path):
    processed_relation_set = []

    with open(file_path, 'r') as file:
        for text in file:
            subject,predicate,obj = extract_words(text)

            subject = word_process(subject)
            predicate = word_process(predicate)
            obj = word_process(obj)

            #print(subject,predicate,obj)
            processed_relation_set.append((subject,predicate,obj))

    #print(processed_relation_set)

    write_relations_secondary(processed_file_path,processed_relation_set)

def write_relations_secondary(file_path,relation_set):
    #print(file_path,"\n\n\n",relation_set)

    with open(file_path,'w') as file:
        for relation in relation_set:
            file.write(str(relation) + "\n")

def write_relations_to_file(file_path,entity_dict):
    with open(file_path,'w') as file:
        for key in entity_dict:
            for value in entity_dict[key]:
                file.write(str(value) + "\n")

def write_relations_secondary(file_path,relation_set):
    #print(file_path,"\n\n\n",relation_set)

    with open(file_path,'w') as file:
        for relation in relation_set:
            file.write(str(relation) + "\n")

def write_entities_to_file(file_path,entity_set):
    with open(file_path,'w') as file:
        for entity in entity_set:
            file.write(entity.strip() + "\n")

def get_entities_from_file(file_path):
    entity_list = []

    with open(file_path, 'r') as file:
        for text in file:
            subject = text[text.find('(') + 1:text.find(',')].replace("'", "")
            entity_list.append(subject.replace("'", ""))

    return entity_list

def get_relations_from_file(file_path):
    relation_list = []

    with open(file_path, 'r') as file:
        for text in file:
            subject,predicate,obj = extract_words(text)

            #print(subject,predicate,obj)
            #entity_set.add(subject.replace("'", ""))
            relation_list.append((subject,predicate,obj))

    return relation_list

def create_relation_dict(relation_list,relation_processed_list):
    relation_dict = dict()

    for idx,relation in enumerate(relation_list):
        relation_dict[relation] = relation_processed_list[idx]

    return relation_dict

def compute_entitiy_similarity(entity_processed,list_entity_processed):
    return fuzz.ratio(entity_processed, list_entity_processed) > 70

def add_to_final_ent_list(gpt_final_list_entity,entity):
    add_entity = True
    entity_processed = word_process(entity.replace("\n",""))

    for list_entity in gpt_final_list_entity:
        list_entity_processed = word_process(list_entity.replace("\n",""))

        if compute_entitiy_similarity(entity_processed,list_entity_processed): 
            add_entity = False
            break

    return add_entity

def filter_entities(gpt_entities_path,gpt_final_list_entity):
    #final_entity_list = []

    with open(gpt_entities_path,'r') as file:
        for entity in file:
            #print(entity.replace("\n",""))
            if add_to_final_ent_list(gpt_final_list_entity,entity):
                #print("REACHED -> ", entity.replace("\n",""))
                gpt_final_list_entity.append(entity.replace("\n",""))

    #print(gpt_entities_path,"\n")
    write_entities_to_file(gpt_entities_path,gpt_final_list_entity)
    
    return len(gpt_final_list_entity)

def create_final_set(prefinal_set,final_set,final_list_entity):
    #print(prefinal_set,"\n\n\n")
    #print(final_list_entity)

    for relation in prefinal_set:
        subject = extract_words(str(relation))[0]

        #print("SUBJECT -> ", subject)

        if subject in final_list_entity:
            #print("REACHED -> subject = ", subject,subject in final_list_entity)
            final_set.add(relation)

    #print(final_set)

def delete_output_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

def filter_outputs_gpt4():
    write_processed_relations(gpt4_outputs_path,gpt4_outputs_processed_path)

    gpt4_entity_list = get_entities_from_file(gpt4_entities_path)
    gpt4_relation_list = get_relations_from_file(gpt4_outputs_path)
    gpt4_relation_processed_list = get_relations_from_file(gpt4_outputs_processed_path)
    gpt4_relation_dict = create_relation_dict(gpt4_relation_list,gpt4_relation_processed_list)

    compute_relation_similarity(gpt4_relation_list,gpt4_relation_dict,gpt4_similarity_dict)
    filter_list(gpt4_similarity_dict,gpt4_relation_list,gpt4_prefinal_set)
    create_final_set(gpt4_prefinal_set,gpt4_final_set,gpt4_final_list_entity)
    write_relations_secondary(gpt4_outputs_path,gpt4_final_set)

    delete_output_file(gpt4_outputs_processed_path)

    return len(gpt4_final_set)

def filter_output_pronouns(file_path):
    filtered_entities_set = set()

    with open(file_path, 'r') as file:
        for text in file:
            subject = text[text.find('(') + 1:text.find(',')].replace("'", "")
            filtered_entities_set.add(subject.replace("'", ""))

    return filtered_entities_set

def tf_idf_vectorizer(input_string,extracted_text):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(extracted_text + [input_string])
    feature_names = vectorizer.get_feature_names_out()

    input_tfidf_scores = tfidf_matrix[-1].toarray()[0]
    average_tfidf_score = sum(input_tfidf_scores) / len(extracted_text)

    #print("TF_IDF_SCORE -> ", average_tfidf_score, " | ENTITY -> ", input_string)

    return average_tfidf_score

def filter_output_relations(entity_set,file_path,extracted_text):
    filtered_entities_dict = dict()
    
    for entity in entity_set:
        relations_set = get_relations(entity,file_path)
        score = 0

        if relations_set:
            score += len(relations_set)
            score += tf_idf_vectorizer(entity,extracted_text)

            filtered_entities_dict[entity] = (score,relations_set)

            #print(entity,relations_set)

    return filtered_entities_dict    

def process_verb_predicate(predicate,full_triple):
    doc = nlp(predicate)
    processed_tokens = []

    predicate_verb_set = set()

    for token in doc:
        #print(token,token.pos_,token.dep_)
        if token.pos_.lower() == 'verb' and token.tag_.lower() != 'vbg':
            if token.lemma_ in lemmatized_verb_dict:
                lemmatized_verb_dict[token.lemma_] += 1
            else:
                lemmatized_verb_dict[token.lemma_] = 1

            processed_tokens.append(token.lemma_)
            predicate_verb_set.add(token.lemma_)
        elif token.pos_.lower() == 'noun' or token.pos_.lower() == 'adj': #or token.dep_ == 'neg':
            processed_tokens.append(token.text)

    return (' '.join(processed_tokens),predicate_verb_set)
    #print("PREDICATE -> ", predicate, "<-> ", ' '.join(processed_tokens))

def expand_acronym(subject):
    return_subject = subject

    if subject in gdc_acronyms:
        #print("EXPANDED BECOMES -> ", gdc_acronym_dict[subject])
        return_subject = gdc_acronym_dict[subject]

    if subject in helio_acronyms_set:
        #print("EXPANDED BECOMES -> ", helio_acronyms_set_dict[subject])
        return_subject = helio_acronyms_set_dict[subject]

    return return_subject
    #print("**********>>>")

def entity_linking(subject):
    #print("SUBJECT -> ", subject)
    doc = nlp(subject)
    linked_entities_collection = doc._.linkedEntities

    for linkage in linked_entities_collection:
        if linkage.get_id() in entity_linkage_dict:
            entity_linkage_dict[(linkage.get_id(),linkage.get_description())].add(subject)
        else:
            entity_linkage_dict[(linkage.get_id(),linkage.get_description())] = {subject}

def word_sense_disambiguation(subject,complete_triple):
    lesk_output = adapted_lesk(complete_triple,subject)
    
    if lesk_output:
        #print("SUBJECT -> ", subject, " COMPLETE_TRIPLE -> ", complete_triple)
        #print(lesk_output.definition())

        if lesk_output.definition() in entity_disambiguation_dict:
            entity_disambiguation_dict[lesk_output.definition()].add(subject)
        else:
            entity_disambiguation_dict[lesk_output.definition()] = {subject}

def process_subject_entity(subject,complete_triple):
    #print("<<<<***")
    subject_processed = word_process(subject)

    highest_score = -1
    matched_entity = str()

    for processed_entity in glossary_dict_processed:
        if fuzz.ratio(subject_processed,processed_entity) > 90:
            original_entity = glossary_dict_processed[processed_entity]
            score = fuzz.ratio(subject_processed,processed_entity)

            if highest_score < score: 
                highest_score = score
                matched_entity = original_entity

            #print("SUBJECT -> ", subject_processed, " ENTITY -> ", processed_entity, " SCORE -> ", fuzz.ratio(subject_processed,processed_entity))
            #print("ORIGINAL SUBJECT -> ", subject, "ORIGINAL ENTITY -> ", original_entity," SCORE -> ", fuzz.ratio(subject_processed,processed_entity),"\n\n")

    if highest_score > -1:
        #print("IN GLOSSARY : FINAL MATCH -> ",subject,"<->",matched_entity,highest_score)
        return (expand_acronym(matched_entity),False)
    else:
        #entity_linking(subject)
        #word_sense_disambiguation(subject,complete_triple)
        #print(subject," NOT IN GLOSSARY")
    
        return (subject,True)
    #print("***>>>")

def find_key_by_value_in_set(dictionary, target_value):
    for key, value_set in dictionary.items():
        if target_value in value_set:
            return key
    return None

def final_triple_postprocess(triple_postprocess_list):
    final_output_dict = dict()

    final_ent_list = []
    final_triples_list = []
    
    for triple in triple_postprocess_list:
        predicate_verb_set = triple[1][1]
        #print(triple)

        if len(predicate_verb_set) >= 1:
            #found_key = find_key_by_value_in_set(my_dict, value_to_find)
            #print(triple,"\n")

            entity = triple[0][0]
            predicate = triple[1][0]
            obj = triple[2]

            semantic_triple = (entity,predicate,obj)
            #print(entity,"|",predicate,"|",obj)

            if entity in final_output_dict:
                final_output_dict[entity].add(semantic_triple)
            else:
                final_output_dict[entity] = {semantic_triple}

    for key in final_output_dict:
        if len(final_output_dict[key]) > 1:
            final_ent_list.append(key)
            
            for triple in final_output_dict[key]:
                final_triples_list.append(triple)

    write_relations_secondary(gpt4_graph_entities_path,final_ent_list)
    write_relations_secondary(gpt4_graph_outputs_path,final_triples_list)
    #print(key,final_output_dict[key],"\n")

def create_graph_triples_outputs(gpt4_outputs_path):
    #with open(gpt4_graph_outputs_path,'a') as file:
    #            file.write(str(semantic_triple) + "\n")

    triple_postprocess_list = []

    with open(gpt4_outputs_path,'r') as file:
        for line in file:
            subject,predicate,obj = extract_words(line)
            full_triple = subject + ' ' + predicate + ' ' + obj

            processed_predicate = process_verb_predicate(predicate,full_triple)
            #print(full_triple,"|",predicate,"|",processed_predicate)
            processed_subject = process_subject_entity(subject,full_triple)
            
            #semantic_triple = (processed_subject,processed_predicate,obj)

            #print(processed_subject,"|",processed_predicate,"|",obj)

            triple_postprocess_list.append((processed_subject,processed_predicate,obj))
            #if processed_subject:
            #    print(subject,"|",processed_subject)

        #print(lemmatized_verb_dict)
        #print_dictionary(entity_linkage_dict)
        #print_dictionary(entity_disambiguation_dict)

    final_triple_postprocess(triple_postprocess_list)

def print_dictionary(dictionary):
    for key in dictionary:
        print(key,dictionary[key])
    
    print("**********************")

def create_graph_text_outputs(gpt4_outputs_path):
    acronym_set_curation()
    curate_yamz_set()
    #consolidate_acronyms(gdc_report_path,page_number)

    consolidate_glossary_set()
    #print(len(glossary_set))

    create_graph_triples_outputs(gpt4_outputs_path)
    
def get_relations(entity,file_path):
    relations_set = set()

    with open(file_path, 'r') as file:
        for line in file:
            subject = extract_words(str(line))[0]
            
            if entity == subject: 
                relations_set.add(line.strip().rstrip("\n"))

                #print(entity,line)

    return relations_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF Text List")
    parser.add_argument("pdf_text_list", nargs="+", help="List of extracted text from the PDF")
    args = parser.parse_args()

    extracted_text = args.pdf_text_list

    #print(extracted_text)

    #print("REACHED")
    #delete_output_file(gpt4_entities_path)
    #delete_output_file(gpt4_outputs_path)
    
    process_extracted_text_gpt(extracted_text,model_gpt4,gpt4_entities_path,gpt4_outputs_path)
    
    num_entities = filter_entities(gpt4_entities_path,gpt4_final_list_entity)
    num_outputs = filter_outputs_gpt4()

    print(num_entities, "entities saved to", gpt4_entities_path)
    print(num_outputs, "semantic triple(s) saved to", gpt4_outputs_path)   

    create_graph_text_outputs(gpt4_outputs_path)

    filtered_entities_set = filter_output_pronouns(gpt4_graph_entities_path)
    #print(filtered_entities_set)

    filtered_entities_dict = filter_output_relations(filtered_entities_set,gpt4_graph_outputs_path,extracted_text)
    #print("******** -> ", filtered_entities_dict)
    
    #filtered_entities_dict = {'Small-scale features': (2.3411013502433096, {"('Small-scale features', 'play important roles', 'creation and behavior in multi-scale dynamics')", "('Small-scale features', 'have net effects', 'large-scale dynamics')"})}

    delete_output_file(gpt4_graph_entities_path)
    delete_output_file(gpt4_graph_outputs_path)

    #delete_output_file(gpt4_entities_path)
    #delete_output_file(gpt4_outputs_path)

    if not filtered_entities_dict:
        raise Exception("Dictionary is empty, cannot proceed to the next script")

    create_outputs_path = "./create_graph_outputs.py"
    subprocess.run(["python", create_outputs_path,str(filtered_entities_dict)])