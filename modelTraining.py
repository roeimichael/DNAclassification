import warnings
from CNN2D import DNA_CNN
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch

warnings.filterwarnings("ignore")


def load_data():
    reduced_data = pd.read_csv("./data/reduced_data_encoded.csv")
    sequences = reduced_data["encoding"].tolist()
    labels = reduced_data["label"].values
    ids = reduced_data["id"].values

    sequences = [[int(digit) for digit in sequence] for sequence in sequences]  # Convert string sequences to lists of integers

    sequences = torch.LongTensor(sequences)
    labels = torch.LongTensor(labels)

    # Adjust sequence lengths
    seq_lengths = [len(seq) for seq in sequences]
    max_length = max(seq_lengths)
    min_length = min(seq_lengths)
    target_length = (max_length + min_length) // 2
    for i in tqdm(range(len(sequences)), desc="Processing sequences"):
        difference = target_length - len(sequences[i])
        if difference > 0:  # Sequence is shorter than target length
            padding = [[0, 0] for _ in range(difference)]
            sequences[i] = torch.cat((sequences[i], torch.LongTensor(padding)), dim=0)
        elif difference < 0:  # Sequence is longer than target length
            sequences[i] = sequences[i][:target_length]

    return sequences, labels, idss






def train(model, criterion, optimizer, train_loader, device, num_epochs, num_classes):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)  # Pass inputs to the model

            targets = targets.view(-1)
            outputs = outputs.view(-1, num_classes)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()

            if batch_idx % 10 == 9:
                batch_loss = running_loss / 10
                accuracy = 100 * correct_predictions / (10 * len(inputs))
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] - Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {batch_loss:.4f} - Accuracy: {accuracy:.2f}%")
                running_loss = 0.0
                correct_predictions = 0


def evaluate(model, test_loader, device):
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


def compare_labels_ids(labels_csv, ids_csv):
    reduced_data = pd.read_csv("./data/reduced_data_encoded.csv")
    reduced_data["id"] = reduced_data["id"].str.strip('"')

    csv_dict = dict(zip(ids_csv, labels_csv))

    for id in ids_csv:
        label_csv = csv_dict[id]
        label_reduced = reduced_data.loc[reduced_data["id"] == id, "label"].values[0]
        if label_csv != label_reduced:
            print(f"Mismatch for ID: {id}")
            print(f"Label from CSV file: {label_csv}")
            print(f"Label from reduced_data_encoded.csv file: {label_reduced}")
            print()


def main():
    sequences, labels, ids = load_data()

    print(sequences.shape)
    print(labels.shape)

    sequences = sequences.unsqueeze(1)
    labels = labels.squeeze()

    compare_labels_ids(labels, ids)

    print(sequences.shape)
    print(labels.shape)

    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = len(labels.unique())
    model = DNA_CNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    avg = train(model, criterion, optimizer, train_loader, device, num_epochs=10, num_classes=num_classes)
    print(avg)

    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=32)
    predicted, test_labels = evaluate(model, test_loader, device)

    print(classification_report(test_labels.cpu().numpy(), predicted.cpu().numpy()))


if __name__ == "__main__":
    main()
