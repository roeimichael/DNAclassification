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
samples_list = []
total_files_processed = 0

for line in metadata_lines[1:]:
    columns = line.strip().split("\t")
    accession_id = columns[0]
    lineage = columns[1]
    lineage_samples[lineage].append(accession_id)
    unique_lineages.add(lineage)

selected_lineages = [lineage for lineage in unique_lineages if len(lineage_samples[lineage]) >= 2000][:25]
max_sequences_per_lineage = 200
max_sequences = max_sequences_per_lineage * len(selected_lineages)


def download_sample(sample, lineage):
    url = f"https://www.ebi.ac.uk/Tools/dbfetch/dbfetch?db=ena_sequence&id={sample}&format=fasta&style=raw&Retrieve=Retrieve"
    response = requests.get(url)
    if response.status_code == 200:
        filename = re.sub(r'[\\/:\*\?"<>\|()]', '_', sample)
        sequence = str(list(SeqIO.read(io.StringIO(response.text), "fasta").seq))
        with io.open(f"./data/fasta_files/{filename}.fasta", "w", encoding="utf-8") as seq_file:
            seq_file.write(f'>{filename}\n{sequence}')
        return {'id': sample, 'lineage': lineage}


with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    for lineage in tqdm(selected_lineages, desc="Processing lineages"):
        samples = random.sample(lineage_samples[lineage], max_sequences_per_lineage)
        lineage_samples[lineage] = list(set(lineage_samples[lineage]) - set(samples))
        futures = {executor.submit(download_sample, sample, lineage) for sample in samples}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                samples_list.append(result)
                total_files_processed += 1
                if total_files_processed >= 5000:
                    break
        if total_files_processed >= 5000:
            break

print("Download completed!")

# Convert the list of dictionaries to a dataframe
df = pd.DataFrame(samples_list)

# Save the dataframe to csv
df.to_csv("./data/dataset.csv", index=False)
