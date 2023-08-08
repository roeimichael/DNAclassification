# import os
# import random
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from tqdm import tqdm
#
#
# def read_fasta_file(path):
#     with open(path, "r") as file:
#         lines = file.readlines()
#         sequence = ''.join(lines[1:]).replace('\n', '')
#         sequence = ''.join(c for c in sequence if c.isalpha())  # This will remove non-alpha characters
#         return sequence
#
#
# def encode_sequence(sequence):
#     encoded_sequence = ""
#     # Create a dictionary for additional characters
#     char_dict = {'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'], 'W': ['A', 'T'],
#                  'K': ['G', 'T'], 'M': ['A', 'C'], 'B': ['C', 'G', 'T'],
#                  'D': ['A', 'G', 'T'], 'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'N': ['A']}
#     for char in sequence:
#         if char in ('A', 'C', 'G', 'T'):
#             if char == 'A':
#                 encoded_sequence += "00"
#             elif char == 'C':
#                 encoded_sequence += "01"
#             elif char == 'G':
#                 encoded_sequence += "10"
#             elif char == 'T':
#                 encoded_sequence += "11"
#         elif char in char_dict.keys():
#             selected_char = random.choice(char_dict[char])
#             if selected_char == 'A':
#                 encoded_sequence += "00"
#             elif selected_char == 'C':
#                 encoded_sequence += "01"
#             elif selected_char == 'G':
#                 encoded_sequence += "10"
#             elif selected_char == 'T':
#                 encoded_sequence += "11"
#     return encoded_sequence
#
#
# def process_sequence(sequence):
#     return encode_sequence(sequence)
#
#
# reduced_data = pd.read_csv("./data/dataset.csv")
# fasta_files = os.listdir("./data/fasta_files/")
#
# sequence_lengths = []  # List to store sequence lengths
#
# for fasta_file in tqdm(fasta_files, desc="Processing fasta files"):
#     id = fasta_file.split("_")[1]
#     sequence = read_fasta_file(f"./data/fasta_files/{fasta_file}")
#     encoded_sequence = process_sequence(sequence)
#
#     with open(f'./data/encoded_sequences/{id}.txt', 'w') as file:
#         file.write(encoded_sequence)
#
#     sequence_lengths.append(len(encoded_sequence))  # Add the length of the sequence to the list
#
# # Assuming that reduced_data dataframe has the same number of rows and in the same order as fasta_files
# reduced_data['sequence_length'] = sequence_lengths  # Add the list as a new column to the dataframe
#
# reduced_data.to_csv("./data/dataset.csv", index=False)
#
import os
import random
from tqdm import tqdm


def read_fasta_file(path):
    with open(path, "r") as file:
        lines = file.readlines()
        sequence = ''.join(lines[1:]).replace('\n', '')
        sequence = ''.join(c for c in sequence if c.isalpha())  # This will remove non-alpha characters
        return sequence


def encode_sequence(sequence):
    encoded_sequence = ""
    char_dict = {'R': ['A', 'G'], 'Y': ['C', 'T'], 'S': ['G', 'C'], 'W': ['A', 'T'],
                 'K': ['G', 'T'], 'M': ['A', 'C'], 'B': ['C', 'G', 'T'],
                 'D': ['A', 'G', 'T'], 'H': ['A', 'C', 'T'], 'V': ['A', 'C', 'G'], 'N': ['A']}
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
        elif char in char_dict.keys():
            selected_char = random.choice(char_dict[char])
            if selected_char == 'A':
                encoded_sequence += "00"
            elif selected_char == 'C':
                encoded_sequence += "01"
            elif selected_char == 'G':
                encoded_sequence += "10"
            elif selected_char == 'T':
                encoded_sequence += "11"
    return encoded_sequence


def process_sequence(sequence):
    return encode_sequence(sequence)


# Get all the directories within fasta_files directory
all_directories = [d for d in os.listdir("./data/fasta_files") if os.path.isdir(os.path.join("./data/fasta_files", d))]

for lineage in tqdm(all_directories, desc="Processing lineages"):
    lineage_dir = f"./data/fasta_files/{lineage}"
    encoded_lineage_dir = f"./data/encoded_sequences/{lineage}"

    # Create corresponding lineage folder in encoded_sequences if it doesn't exist
    if not os.path.exists(encoded_lineage_dir):
        os.makedirs(encoded_lineage_dir)

    for filename in os.listdir(lineage_dir):
        fasta_file_path = os.path.join(lineage_dir, filename)
        sequence = read_fasta_file(fasta_file_path)
        encoded_sequence = process_sequence(sequence)

        # Save encoded sequence in corresponding lineage folder
        with open(os.path.join(encoded_lineage_dir, filename.replace(".fasta", ".txt")), 'w') as file:
            file.write(encoded_sequence)

print("Encoding completed!")
