import argparse
import subprocess

import os

import csv
import re

import shutil

acronyms_dict = {}
replacements = {}

pattern = r'[^a-zA-Z0-9\s]'

def curate_dictionary():
	with open('solar_physics_acronyms.csv', newline='') as csvfile:
	    reader = csv.DictReader(csvfile)
	    for row in reader:
	        acronyms_dict[row['acronyms']] = row['terms'].strip()

def split_triple(orig_triple):
	subject_entity = orig_triple[0]
	subject_entity = re.sub(pattern, '',subject_entity)

	predicate_entity = orig_triple[1]
	predicate_entity = re.sub(pattern, '',predicate_entity)

	object_entity = orig_triple[2]
	object_entity = re.sub(pattern, '',object_entity)

	"""
	print("SUBJECT ENTITY -> ", subject_entity)
	print("PREDICATE ENTITY -> ", predicate_entity)
	print("OBJECT ENTITY -> ", object_entity,"*****************\n")
	"""

	subject_entity_split = subject_entity.split()
	predicate_entity_split_ = predicate_entity.split()
	object_entity_split = object_entity.split()

	"""
	print("SUBJECT ENTITY SPLIT -> ", subject_entity_split)
	print("PREDICATE ENTITY SPLIT -> ", predicate_entity_split_)
	print("OBJECT ENTITY SPLIT -> ", object_entity_split,"\n\n\n")
	"""

	return (subject_entity_split,predicate_entity_split_,object_entity_split)

def replace_acronym(split_entity):
	return_arr = []
	flag = False

	for word in split_entity:
		if word in acronyms_dict:
			#flag = True
			"""
			print("***********WORD -> ", word, " SPLIT ENTITY -> ", split_entity, "\n\n\n")
			print("expanded_form -> ", expanded_form)
			"""
			expanded_form = acronyms_dict[word]
			return_arr.append(expanded_form)

			replacements[word] = replacements.get(word,0) + 1
		else:
			return_arr.append(word)

	if flag:
		print("RETURN ARR -> ", return_arr)
		print("SPLIT ENTITY -> ", split_entity,"\n\n\n")

		print("RETURN STR -> ", ' '.join(return_arr))
		print("SPLIT ENTITY STR -> ", ' '.join(split_entity),"\n\n\n")

	return ' '.join(return_arr)

def process_sentences(input_file,acronyms_dict):
	temp_file = input_file + '.temp'

	with open(input_file, 'r') as infile, open(temp_file, 'w') as outfile:
		print("REACHED")
		for line in infile:
			orig_triple = eval(line.strip())

			"""
			subject_entity_prev = orig_triple[0]
			subject_entity = re.sub(pattern, '',subject_entity_prev)

			expanded_form = None
			"""

			"""
			if subject_entity in acronyms_dict:
				#print("SUBJECT ENTITY IS DIRECT MATCH IN DICT", subject_entity)
				expanded_form = acronyms_dict[subject_entity_prev]
				orig_triple = (expanded_form,) + orig_triple[1:]
				replacements[subject_entity] = replacements.get(subject_entity,0) + 1

				#print("ORIG TRIPLE IS NOW -> ", orig_triple)
			else :
			"""

			resultant_splits = split_triple(orig_triple)

			print("resultant_splits -> ", resultant_splits," LEN SPLIT -> ", len(resultant_splits))
			return_triple = ()
			join_arr = []

			for arr in resultant_splits:
				return_str = replace_acronym(arr)

				join_arr.append(return_str)
				"""
				if(input_str != return_str):
					print("INPUT STR -> ", input_str)
					print("RETURN STR -> ", return_str,"\n\n\n")

					join_arr.append(return_str)
				"""

			append_tuple = '(' + ','.join([f"'{s}'" for s in join_arr]) + ')'
			append_str = str(append_tuple)

			print("APPEND_STR -> ", append_str)

			modified_line = append_str + '\n'
			outfile.write(modified_line)

	shutil.move(temp_file,input_file)

	with open('acronyms_replacement.txt','w') as txtfile:
		for acronym,count in replacements.items():
			txtfile.write(f"{acronym} -> {count}\n")

if __name__ == "__main__":
	print("REACHED ACRONYM RESOLVER")
	curate_dictionary()

	#print(acronyms_dict)
	process_sentences("first_triples.txt",acronyms_dict)

	resolve_synonyms_path = "./ads_synonym_resolver.py"
	subprocess.run(["python",resolve_synonyms_path])
