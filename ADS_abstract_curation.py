import os
import re

# Create subfolder if it doesn't exist

subfolder = "ADS_abstracts"

if not os.path.exists(subfolder):
    os.makedirs(subfolder)

# Read the contents of the file
with open("export-rss_MI_coupling.rss", "r") as file:
    content = file.read()

# Find and extract abstracts
abstracts = re.findall(r'<description>(.*?)</description>', content, re.DOTALL)
print("Number of abstracts:", len(abstracts))

# Extract filenames from links
filenames = re.findall(r'<link>https://ui.adsabs.harvard.edu/abs/([^<]+)</link>', content)
print("Number of filenames:", len(filenames))

# Write abstracts to separate files
for filename, abstract in zip(filenames, abstracts):
    filename = os.path.join(subfolder, filename + ".txt")
    with open(filename, "w") as file:
        file.write(abstract)