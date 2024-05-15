import os
import subprocess
import argparse

#folder_path = "./papers"
pdf_processing_script = "process_pdf.py"  

def process(pdf_directory,output_directory):
	file_names = []

	for root, dirs, files in os.walk(pdf_directory):
	    for file in files:   
	    	#print(file) 	
	    	file_names.append(file)

	print("FILE NAMES -> ", file_names)

	for file_name in file_names:
		print(file_name)
		command = ["python", pdf_processing_script, os.path.join(pdf_directory, file_name), os.path.splitext(file_name)[0], output_directory]

		# Call the PDF processing script using subprocess
		try:
		    subprocess.run(command, check=True)
		    print(f"PDF processing completed for {file_name}")
		except subprocess.CalledProcessError as e:
		    print(f"Error processing PDF for {file_name}: {e}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="PDF Processing Script")
	parser.add_argument("pdf_directory", help="Directory to read PDFs from")
	parser.add_argument("output_directory", help="Directory to store outputs in")
	args = parser.parse_args()

	if args.pdf_directory == "<pdf_directory>":
		print("Please provide directory to read inputs from")
	elif args.output_directory == "<output_directory>":
		print("Please provide directory to store outputs in")
	else:
		try:
			pdf_directory = args.pdf_directory
			output_directory = args.output_directory

			process(pdf_directory,output_directory)
		except Exception as e:
			print(f"Error processing: {e}")