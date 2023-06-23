import os
import random
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def read_fasta_file(path):
    with open(path, "r") as file:
        lines = file.readlines()
        sequence = ''.join(lines[1:]).replace('\n', '')
        return sequence

def encode_sequence(sequence):
    encoded_sequence = ""
    for char in sequence:
        if char in ('A', 'C', 'G', 'T'):
            if char == 'A':
                encoded_sequence += "00"
            elif char == 'C':
                encoded_sequence += "01"
            elif char == 'G':
                encoded_sequence += "10"
            elif char == 'T':
                encoded_sequence += "11"
        else:
            random_encoding = random.choice(["00", "01", "10", "11"])
            encoded_sequence += random_encoding
    return encoded_sequence

reduced_data = pd.read_csv("./data/dataset.csv")
fasta_files = os.listdir("./data/fasta_files/")

encoded_sequences = []  # List to store encoded sequences
sequences = []
for fasta_file in tqdm(fasta_files, desc="Processing fasta files"):
    id = fasta_file.split("_")[1]
    sequence = read_fasta_file(f"./data/fasta_files/{fasta_file}")
    encoded_sequence = encode_sequence(sequence)
    sequences.append(sequence)
    encoded_sequences.append(encoded_sequence)

reduced_data["encoding"] = encoded_sequences
reduced_data["sequence"] = sequences

label_encoder = LabelEncoder()
reduced_data["label"] = label_encoder.fit_transform(reduced_data["lineage"])

reduced_data.to_csv("./data/reduced_data_encoded.csv", index=False)
