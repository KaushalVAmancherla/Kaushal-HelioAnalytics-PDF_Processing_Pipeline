import pandas as pd
import re

import argparse
import subprocess

import openpyxl

import spacy
import nltk
from nltk.stem import WordNetLemmatizer

import os

nltk.download('wordnet')

nlp = spacy.load("en_core_web_md")
lemmatizer = WordNetLemmatizer()

pattern = r'[^a-zA-Z0-9\s]'

synonym_file_path = './ADS_synonym_data/ads_simple_synonyms_updated.txt'
synonym_file_path_second = './ADS_synonym_data/ads_OpenAlex_synonyms_updated_v4.txt'
synonym_file_path_third = './ADS_synonym_data/AGU-index-terms-synonyms-workingList.xlsx'

sentences_file_path = 'first_triples.txt'
modified_sentences_file_path = 'final_triples.txt'
replacement_stats_file_path = 'replacement_stats.txt'

synonym_dict = {}
replacement_stats = {}

def append_to_dict(synonyms):
    found = False

    keys_to_modify = []
    synonyms_lemmatized_check = []

    for key in synonym_dict:
        if any(word in key for word in synonyms):
            #print("ENTRY IN SYNONYM LIST ALREADY EXISTS AS KEY IN DICTIONARY")
            #print(synonyms, " => OG KEY ", key)

            # Add the key to the list of keys to be modified
            keys_to_modify.append(key)
            found = True
            break

    for key in keys_to_modify:
        #lemmatized_value = lemmatizer.lemmatize(synonym_dict[key])
        lemmatized_value = lemmatize_phrase(synonym_dict[key])

        #print("LEMATIZED VALUE => ", lemmatized_value, " ORIGINAL VALUE => ", synonym_dict[key])

        for word in synonyms:
            lemmatized_word = lemmatize_phrase(word)

            if lemmatized_word != lemmatized_value:
                print("WORD -> ", word, "VALUE -> ", synonym_dict[key], " LEMMATIZED WORD ->", lemmatized_word, " LEMMATIZED VALUE -> ", lemmatized_value)
                synonyms_lemmatized_check.append(word)

        # Combine the existing key and synonyms, avoiding duplicates
        new_key = tuple(set(key) | set(synonyms_lemmatized_check))
        #print("NEW KEY -> ", new_key)
        
        # Update dictionary with the new key and remove the old key
        synonym_dict[new_key] = synonym_dict.pop(key)

    return found

def lemmatize_phrase(phrase):
    # Split the phrase into words

    words = phrase.split()

    # Lemmatize each word and join them back into a phrase
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_phrase = ' '.join(lemmatized_words)

    """
    doc = nlp(phrase)
    lemmatized_phrase = ' '.join([token.lemma_ for token in doc])
    """

    return lemmatized_phrase

def read_synonym_mappings(file_path,second_file,third_file):   
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                synonyms_list, central_word = line.split(' => ')

                synonyms = synonyms_list.split(', ')
                synonyms_re_sub = [re.sub(pattern, '', word.lower()) for word in synonyms]
                central_word_re_sub = re.sub(pattern, '', central_word).lower()
                synonyms_lemmatized_check = []

                #central_word_re_sub_lemma = lemmatizer.lemmatize(central_word_re_sub)
                central_word_re_sub_lemma = lemmatize_phrase(central_word_re_sub)

                for word in synonyms_re_sub:
                    lemmatized_word = lemmatize_phrase(word)

                    if lemmatized_word != central_word_re_sub_lemma:
                        synonyms_lemmatized_check.append(word)
                        print("WORD -> ", word, "VALUE -> ", central_word_re_sub, " LEMMATIZED WORD ->", lemmatized_word, " LEMMATIZED VALUE -> ", central_word_re_sub_lemma)

                    """
                    lemmatized_word = doc[0].lemma_  
                    synonyms_lemmatized.append(lemmatized_word)
                    """

                synonym_dict[tuple(element.lower() for element in synonyms_lemmatized_check)] = central_word_re_sub

    with open(second_file, 'r') as file:
        for line in file:
            line = line.strip()

            if line:
                synonyms = []    
                parts = line.split(';')

                for part in parts:
                    part = part.strip()
                    part_re = re.sub(pattern, '', part).lower()
                    synonyms.append(part_re)
            
                if not append_to_dict(synonyms):
                    #print("ADDED NEW ENTRY -> ", tuple(synonyms[:-1]), " => ", synonyms[-1])
                    synonym_dict[tuple(synonyms[:-1])] = synonyms[-1]

                """
                for key in synonym_dict:
                    if any(word in key for word in synonyms):
                        print("ENTRY IN SYNONYM LIST ALREADY EXISTS AS KEY IN DICTIONARY")

                        new_key = tuple(set(key) | set(synonyms))
                        print("NEW KEY -> ", new_key)
                        
                        # Update dictionary with the new key and remove the old key
                        synonym_dict[new_key] = synonym_dict.pop(key)
                        found = True
                        break
                
                    if not found:
                        #ADD NEW ENTRY INTO SYNONYM DICT, BUT WHAT IS THE KEY AND WHAT IS THE VALUE?
                        synonym_dict[tuple(synonyms[:-1])] = synonyms[-1]
                """
    
    with open(third_file, 'r') as file:
        workbook = openpyxl.load_workbook(third_file)
        terms = []
        synonyms = []

        worksheet = workbook.active

        # Iterate over rows in the worksheet
        for row in worksheet.iter_rows(min_row=2, values_only=True):  # Assuming row 1 contains headers
            term, synonym = row[0], row[1]  # Assuming 'term' is in column A and 'synonyms' in column B
            if term and synonym and "added" not in synonym and "duplicate" not in synonym:
                synonyms = []    
                parts = synonym.split(',')

                for part in parts:
                    part = part.strip()
                    part_re = re.sub(pattern, '', part).lower()
                    synonyms.append(part_re)

                term_re_sub = re.sub(pattern, '', term).lower()

                if not append_to_dict(synonyms):
                    #print("ADDED NEW ENTRY -> ", tuple(synonyms[:-1]), " => ", synonyms[-1])
                    synonym_dict[tuple(synonyms)] = term_re_sub

                """
                terms.append(term)
                synonyms.append(synonym)
                """

        # Now you have the data from 'term' and 'synonyms' columns in the lists 'terms' and 'synonyms'
        #print("Terms:", terms)
        #print("Synonyms:", synonyms)

        """
        for synonym in synonyms:
            if synonym and "added" not in synonym and "duplicate" not in synonym:
                print(synonym)
        """

        term_synonym_dict = {term: synonym for term, synonym in zip(terms, synonyms)}

    return synonym_dict

def split_triple(orig_triple):
    subject_entity = orig_triple[0]
    subject_entity = re.sub(pattern, '',subject_entity)

    predicate_entity = orig_triple[1]
    predicate_entity = re.sub(pattern, '',predicate_entity)

    object_entity = orig_triple[2]
    object_entity = re.sub(pattern, '',object_entity)

    subject_entity_split = subject_entity.split()
    predicate_entity_split_ = predicate_entity.split()
    object_entity_split = object_entity.split()

    return (subject_entity_split,predicate_entity_split_,object_entity_split)

def replace_synonym(split_entity):
    return_arr = []
    flag = False

    for word in split_entity:
        word_processed = re.sub(pattern, '',word.lower())
        append_word = word

        for synonyms, central_word in synonym_dict.items():
            for synonym in synonyms:
                if word_processed == synonym:                
                    linked_word = synonym_dict[synonyms]
                    append_word = linked_word

                    print("PROCESSED WORD -> ", word_processed, " IS IN SYNONYMS -> ", synonyms, " REPLACE WITH -> ", linked_word)

                    if linked_word in replacement_stats:
                        replacement_stats[central_word][0] += 1
                        replacement_stats[central_word][1].append(word)
                    else:
                        replacement_stats[central_word] = [1, [word]]

                    break

        print("APPENDING TO ARR -> ", word)
        return_arr.append(append_word)

    return_str = ' '.join(return_arr)

    print("RETURN_STR => ", return_str)
    return return_str

def process_sentences(file_path, synonym_dict):
    #print("REACHED")
    modified_sentences = []

    #i = 0

    with open(file_path, 'r') as file:
        for line in file:
            print("LINE -> ", line)
            orig_triple = eval(line.strip())
            resultant_splits = split_triple(orig_triple)

            print("RESULTANT SPLITS -> ", resultant_splits)

            return_triple = ()
            join_arr = []

            for arr in resultant_splits:
                return_str = replace_synonym(arr)
                join_arr.append(return_str)

            append_tuple = '(' + ','.join([f"'{s}'" for s in join_arr]) + ')'
            append_str = str(append_tuple)

            modified_sentences.append(append_str)

    return modified_sentences, replacement_stats

def write_modified_sentences(file_path, modified_sentences):
    with open(file_path, 'w') as file:
        for sentence in modified_sentences:
            file.write(sentence + '\n')

def write_replacement_stats(file_path, replacement_stats):
    with open(file_path, 'w') as file:
        for central_word, (count, synonyms) in replacement_stats.items():
            synonyms_str = ', '.join(synonyms)
            file.write(f"{central_word} -> {count}, {synonyms_str}\n")    

if __name__ == "__main__":
    print("REACHED SYNONYM RESOLVER")

    read_synonym_mappings(synonym_file_path,synonym_file_path_second,synonym_file_path_third)

    """
    for entry in synonym_dict:
        print(entry,synonym_dict[entry],"\n\n\n")
    """

    #print(synonym_dict)

    modified_sentences, replacement_stats = process_sentences(sentences_file_path, synonym_dict)
    write_modified_sentences(modified_sentences_file_path, modified_sentences)
    write_replacement_stats(replacement_stats_file_path, replacement_stats)
    
    os.remove("first_triples.txt")

    extract_nodes = "./extract_nodes.py"
    subprocess.run(["python", extract_nodes])