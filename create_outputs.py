import os
import re

import fitz

import openai
import spacy
import nltk
import tiktoken

import argparse

from PIL import Image
from nltk.tokenize import sent_tokenize
from openie import StanfordOpenIE

from spacy.tokens import Token
from spacy.lang.en.stop_words import STOP_WORDS

from collections import defaultdict
from dotenv import load_dotenv
from fuzzywuzzy import fuzz

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

gpt4_entities_path = "gpt4_entities.txt"
gpt4_outputs_path = "gpt4_outputs.txt"
gpt4_outputs_processed_path = "gpt4_outputs_processed.txt"

stop_words_getter = lambda token: token.is_stop or token.lemma_ in STOP_WORDS
Token.set_extension('is_stop', getter=stop_words_getter)

nlp = spacy.load("en_core_web_sm")

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

num_ents = 0
num_triples = 0

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF Text List")
    parser.add_argument("pdf_text_list", nargs="+", help="List of extracted text from the PDF")
    args = parser.parse_args()

    extracted_text = args.pdf_text_list

    #print(extracted_text)

    #print("REACHED")
    delete_output_file(gpt4_entities_path)
    delete_output_file(gpt4_outputs_path)

    process_extracted_text_gpt(extracted_text,model_gpt4,gpt4_entities_path,gpt4_outputs_path)
    
    num_entities = filter_entities(gpt4_entities_path,gpt4_final_list_entity)
    num_outputs = filter_outputs_gpt4()

    print(num_entities, "entities saved to", gpt4_entities_path)
    print(num_outputs, "semantic triple(s) saved to", gpt4_outputs_path)   