import os
import subprocess

folder_path = "ADS_abstracts"  
pdf_processing_script = "process_ADS_abstract.py"  

files = os.listdir(folder_path)

# Loop through each file in the folder
for file_name in files:
    command = ["python", pdf_processing_script, os.path.join(folder_path, file_name), os.path.splitext(file_name)[0] + "_OUTPUTS"]
    
    # Call the PDF processing script using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"PDF processing completed for {file_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing PDF for {file_name}: {e}")