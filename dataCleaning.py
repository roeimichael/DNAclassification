import os

def remove_capital_N(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sequences = []
    current_sequence = ''
    for line in lines:
        line = line.strip()
        if line.startswith('>'):
            if current_sequence:
                sequences.append(current_sequence)
                current_sequence = ''
        else:
            current_sequence += line

    if current_sequence:
        sequences.append(current_sequence)

    modified_sequences = []
    for sequence in sequences:
        # Remove capital 'N's from the start and end of the sequence
        sequence = sequence.lstrip('N').rstrip('N')

        # Check if any capital 'N's are present in the middle of the sequence
        if 'N' in sequence:
            print(f"Found capital 'N' in the middle of the file: {file_path}")

        modified_sequences.append(sequence)

    # Rewrite the modified sequences back into the file
    new_lines = [lines[0]] + [f"{modified_sequences[i][j:j+80]}" for i in range(len(modified_sequences)) for j in range(0, len(modified_sequences[i]), 80)]
    with open(file_path, 'w') as file:
        file.write('\n'.join(new_lines))

# Directory containing the FASTA files
directory = "./data/fasta_files/"

# Iterate over the files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.fasta') or filename.endswith('.fa'):
        file_path = os.path.join(directory, filename)
        remove_capital_N(file_path)
