import warnings
from CNN2D import DNA_CNN
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import classification_report
from tqdm import tqdm

warnings.filterwarnings("ignore")


def load_data():
    data_path = "./data/processed_data.npz"

    if os.path.exists(data_path):
        # Load the sequences and labels from the .npz file if it already exists
        data = np.load(data_path, allow_pickle=True)
        sequences = torch.Tensor(data['sequences'])
        labels = torch.LongTensor(data['labels'])
    else:
        sequences = []
        labels = []
        reduced_data = pd.read_csv("./data/reduced_data_encoded.csv")
        reduced_data["id"] = reduced_data["id"].str.strip('"')
        npy_files = os.listdir("./data/encoded_sequences/")
        for npy_file in npy_files:
            id = npy_file.split(".")[0]
            sequence = np.load(f"./data/encoded_sequences/{npy_file}")
            label = reduced_data.loc[reduced_data["id"] == id, "label"].values
            sequences.append(sequence)
            labels.append(label)

        # Find the maximum and minimum sequence lengths
        seq_lengths = [len(seq) for seq in sequences]
        max_length = max(seq_lengths)
        min_length = min(seq_lengths)
        target_length = (max_length + min_length) // 2

        # Pad or truncate sequences to match the target length
        for i in range(len(sequences)):
            difference = target_length - len(sequences[i])
            if difference > 0:  # Sequence is shorter than target length
                padding = np.repeat([[0.25, 0.25, 0.25, 0.25]], difference, axis=0)
                sequences[i] = np.concatenate([sequences[i], padding])
            elif difference < 0:  # Sequence is longer than target length
                sequences[i] = sequences[i][:target_length]

        # Save the processed sequences and labels to a .npz file
        np.savez(data_path, sequences=sequences, labels=labels)

        sequences = torch.Tensor(sequences)
        labels = torch.LongTensor(labels)

    return sequences, labels

def train(model, criterion, optimizer, train_loader, device):
    model.train()  # set the model to training mode
    total_loss = 0.0
    for i, (sequences, labels) in enumerate(tqdm(train_loader)):
        sequences = sequences.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss



def evaluate(model, test_loader,device):
    model.eval()
    total = 0
    correct = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc="Evaluating"):
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the test sequences: {} %'.format((correct / total) * 100))
    return torch.tensor(all_predictions), torch.tensor(all_labels)


def main():
    data = np.load("./data/processed_data.npz", allow_pickle=True)
    sequences = torch.Tensor(data['sequences'])
    labels = torch.LongTensor(data['labels'])
    print(sequences.shape)
    print(labels.shape)

    sequences = sequences.unsqueeze(1)
    labels = labels.squeeze()

    print(sequences.shape)
    print(labels.shape)

    # sequences, labels = load_data()
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        print('Running on GPU: ', torch.cuda.get_device_name(0))  # Print the name of the GPU
    else:
        print("Running on CPU")

    model = DNA_CNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    avg = train(model, criterion, optimizer, train_loader, device)
    print(avg)

    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=32)
    predicted, test_labels = evaluate(model, test_loader,device)

    print(classification_report(test_labels.cpu().numpy(), predicted.cpu().numpy()))


if __name__ == "__main__":
    main()
