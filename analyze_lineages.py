import os
import pandas as pd
from Bio import SeqIO
import numpy as np
import re
from collections import Counter
from tqdm import tqdm


def clean_name(name):
    """Clean up lineage names and filenames for safe file creation"""
    return re.sub(r'[\\/:\*\?"<>\|]', '', name)


def get_kmer_counts(sequence, k):
    kmer_counts = Counter()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        # ignore k-mers that contain non-ACGT characters
        if re.search('[^ACGT]', kmer):
            continue
        kmer_counts[kmer] += 1
    total_kmers = sum(kmer_counts.values())
    kmer_freqs = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
    return kmer_freqs


# Get all the directories within fasta_files directory
all_directories = [d for d in os.listdir("./data/fasta_files") if os.path.isdir(os.path.join("./data/fasta_files", d))]

lineage_stats = []

for lineage in tqdm(all_directories, desc="Analyzing lineages"):
    cleaned_lineage = clean_name(lineage)
    lineage_dir = f"./data/fasta_files/{cleaned_lineage}"

    seq_lengths = []
    non_actg_counts = []
    all_kmer_freqs = Counter()

    for filename in os.listdir(lineage_dir):
        fasta_file_path = os.path.join(lineage_dir, filename)
        try:
            # Read the sequence data
            record = SeqIO.read(fasta_file_path, "fasta")
            sequence = str(record.seq)
            seq_length = len(sequence)
            non_actg_count = len([base for base in sequence if base not in 'ACTG'])

            seq_lengths.append(seq_length)
            non_actg_counts.append(non_actg_count)

            # Calculate 4-mer frequencies for this sequence
            kmer_freqs = get_kmer_counts(sequence, 4)
            all_kmer_freqs += Counter(kmer_freqs)
        except Exception as e:
            print(f"An error occurred while processing file {fasta_file_path}. Error: {str(e)}")

    # calculate average 4-mer frequencies
    total_sequences = len(seq_lengths)
    avg_kmer_freqs = {kmer: count / total_sequences for kmer, count in all_kmer_freqs.items()}

    lineage_stat = {
        'lineage': lineage,
        'average_seq_len': np.mean(seq_lengths),
        'median_seq_len': np.median(seq_lengths),
        'std_seq_len': np.std(seq_lengths),
        'average_non_actg': np.mean(non_actg_counts),
        'median_non_actg': np.median(non_actg_counts),
        'std_non_actg': np.std(non_actg_counts)
    }
    lineage_stat.update(avg_kmer_freqs)
    lineage_stats.append(lineage_stat)

# Convert the list of dictionaries into a DataFrame
df = pd.DataFrame(lineage_stats)

# Save the DataFrame to a CSV file
df.to_csv("./data/lineage_analysis.csv", index=False)

print("Analysis completed!")
