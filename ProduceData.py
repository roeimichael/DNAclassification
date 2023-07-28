import random
import requests
import io
import re
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import concurrent.futures
import os
from collections import Counter

# The metadata file
with open("./data/metadata", "r") as file:
    metadata_lines = file.readlines()

lineage_samples = defaultdict(list)
unique_lineages = set()

# Prepare a list to hold all samples across all lineages
all_samples = []

def clean_name(name):
    """Clean up lineage names and filenames for safe file creation"""
    return re.sub(r'[\\/:\*\?"<>\|]', '', name)

for line in metadata_lines[1:]:
    columns = line.strip().split("\t")
    accession_id = columns[0]
    lineage = columns[1]
    # Ignore lines with no alphabetical characters in lineage
    if not any(char.isalpha() for char in lineage):
        continue
    lineage_samples[lineage].append(accession_id)
    unique_lineages.add(lineage)

max_sequences_per_lineage = 250

# Calculate top 200 lineages
lineage_counts = Counter({lineage: len(samples) for lineage, samples in lineage_samples.items()})
top_200_lineages = [lineage for lineage, count in lineage_counts.most_common(200)]

# Create directories for the top 200 lineages
for lineage in top_200_lineages:
    cleaned_lineage = clean_name(lineage)
    lineage_dir = f"./data/fasta_files/{cleaned_lineage}"
    os.makedirs(lineage_dir, exist_ok=True)

# Verify that the directories were created successfully
for lineage in top_200_lineages:
    cleaned_lineage = clean_name(lineage)
    lineage_dir = f"./data/fasta_files/{cleaned_lineage}"
    assert os.path.isdir(lineage_dir), f"Directory was not created: {lineage_dir}"


# define download function
# define download function
def download_sample(sample, lineage):
    try:
        url = f"https://www.ebi.ac.uk/Tools/dbfetch/dbfetch?db=ena_sequence&id={sample}&format=fasta&style=raw&Retrieve=Retrieve"
        response = requests.get(url)
        if response.status_code == 200:
            cleaned_lineage = clean_name(lineage)
            lineage_dir = f"./data/fasta_files/{cleaned_lineage}"
            filename = clean_name(sample)
            fasta_file_path = f"{lineage_dir}/{filename}.fasta"
            with io.open(fasta_file_path, "w", encoding="utf-8") as seq_file:
                seq_file.write(response.text)
            # Read the sequence data
            record = SeqIO.read(fasta_file_path, "fasta")
            sequence = str(record.seq)
            seq_length = len(sequence)
            non_actg_count = len([base for base in sequence if base not in 'ACTG'])
            # Append the data to the list
            return {'id': sample, 'lineage': lineage, 'seq_length': seq_length, 'non_actg_count': non_actg_count}
    except Exception as e:
        print(f"Error while processing {sample}: {e}")
        return None


with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    for lineage in tqdm(top_200_lineages, desc="Processing lineages"):
        # Skip if the lineage doesn't have enough samples
        if len(lineage_samples[lineage]) < max_sequences_per_lineage:
            continue

        samples = random.sample(lineage_samples[lineage], max_sequences_per_lineage)
        lineage_samples[lineage] = list(set(lineage_samples[lineage]) - set(samples))
        futures = {executor.submit(download_sample, sample, lineage) for sample in samples}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                all_samples.append(result)

# Save the samples for all lineages to a DataFrame
df = pd.DataFrame(all_samples)
df.to_csv("./data/dataset.csv", index=False)

print("Download completed!")
