import os
import pickle

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np

def read_fasta_file(path):
    with open(path, "r") as file:
        lines = file.readlines()
        sequence = ''.join(lines[1:]).replace('\n', '')
        return sequence

one_hot_encoding = {'A': [1, 0, 0, 0],
                    'C': [0, 1, 0, 0],
                    'G': [0, 0, 1, 0],
                    'T': [0, 0, 0, 1],
                    'R': [0.5, 0, 0.5, 0],
                    'Y': [0, 0.5, 0, 0.5],
                    'S': [0, 0.5, 0.5, 0],
                    'W': [0.5, 0, 0, 0.5],
                    'K': [0, 0, 0.5, 0.5],
                    'M': [0.5, 0.5, 0, 0],
                    'B': [0, 1/3, 1/3, 1/3],
                    'D': [1/3, 0, 1/3, 1/3],
                    'H': [1/3, 1/3, 0, 1/3],
                    'V': [1/3, 1/3, 1/3, 0],
                    'N': [0.25, 0.25, 0.25, 0.25]}

reduced_data = pd.read_csv("./data/dataset.csv")
fasta_files = os.listdir("./data/fasta_files/")

for fasta_file in tqdm(fasta_files, desc="Processing fasta files"):
    id = fasta_file.split("_")[1]
    sequence = read_fasta_file(f"./data/fasta_files/{fasta_file}")
    encoded_sequence = []
    for char in sequence:
        if char.isalpha() and char in one_hot_encoding:
            encoded_sequence.append(one_hot_encoding[char])
        else:
            if char.isalpha():
                encoded_sequence.append(one_hot_encoding['N'])
    np.save(f"./data/encoded_sequences/{id}.npy", encoded_sequence)
    reduced_data.loc[reduced_data["id"] == id, "sequence"] = sequence


label_encoder = LabelEncoder()
reduced_data["label"] = label_encoder.fit_transform(reduced_data["lineage"])

reduced_data.to_csv("./data/reduced_data_encoded.csv", index=False)
