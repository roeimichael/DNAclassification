import random
import requests
import io
import re
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import concurrent.futures

# The metadata file
with open("./data/metadata", "r") as file:
    metadata_lines = file.readlines()

lineage_samples = defaultdict(list)
unique_lineages = set()

# Prepare a list to hold selected samples
samples_list = []

for line in metadata_lines[1:]:
    columns = line.strip().split("\t")
    accession_id = columns[0]
    lineage = columns[1]
    lineage_samples[lineage].append(accession_id)
    unique_lineages.add(lineage)
# Ensure selected lineages have at least 500 samples
selected_lineages = ['"BA.2.31"', '"AY.36.1"', '"B.1.509"', '"AY.8"', '"P.1.17"', '"AY.37"', '"BA.5.1"', '"P.1.13"', '"B.1.258"',
                     '"BA.1.17"']

max_sequences_per_lineage = 500
max_sequences = max_sequences_per_lineage * len(selected_lineages)


# define download function
def download_sample(sample, lineage):
    url = f"https://www.ebi.ac.uk/Tools/dbfetch/dbfetch?db=ena_sequence&id={sample}&format=fasta&style=raw&Retrieve=Retrieve"
    response = requests.get(url)
    if response.status_code == 200:
        filename = re.sub(r'[\\/:\*\?"<>\|()]', '_', sample)
        with io.open(f"./data/fasta_files/{filename}.fasta", "w", encoding="utf-8") as seq_file:
            seq_file.write(response.text)
        # Append the data to the list without the sequence
        return {'id': sample, 'lineage': lineage}


with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    for lineage in tqdm(selected_lineages, desc="Processing lineages"):
        print(len(lineage_samples[lineage]))
        samples = random.sample(lineage_samples[lineage], max_sequences_per_lineage)
        lineage_samples[lineage] = list(set(lineage_samples[lineage]) - set(samples))
        futures = {executor.submit(download_sample, sample, lineage) for sample in samples}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                samples_list.append(result)

print("Download completed!")

df = pd.DataFrame(samples_list)

df.to_csv("./data/dataset.csv", index=False)
