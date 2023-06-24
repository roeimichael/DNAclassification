from ComplexCNN import ComplexCNN
from SimpleCNN import SimpleCNN
from DecentCNN import DecentCNN
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import torch
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


def load_data():
    data = pd.read_csv("./data/dataset.csv")
    data.set_index('id', inplace=True)  # index the dataframe by IDs for direct lookup
    text_files_folder = "./data/encoded_sequences"

    label_encoder = LabelEncoder()
    data['lineage'] = label_encoder.fit_transform(data['lineage'].values)  # directly encode lineages in dataframe

    sequence_lengths = data["sequence_length"].values
    target_length = sequence_lengths.mean().astype(int)

    converted_data, sequences, labels = [], [], []
    for file_path in tqdm(os.listdir(text_files_folder), desc="Processing text files"):
        with open(os.path.join(text_files_folder, file_path), "r") as file:
            content = file.read().strip()
            difference = target_length - len(content)
            if difference > 0:  # Sequence is shorter than target length
                padding = "0" * difference
                content += padding
            elif difference < 0:  # Sequence is longer than target length
                content = content[:target_length]

            file_name = '"' + os.path.splitext(file_path)[0] + '"'
            converted_data.append([file_name, "".join(map(str, content))])

    converted_df = pd.DataFrame(converted_data, columns=["ID", "Enc_Sequence"])  # Create the dataframe
    converted_df["lineage"] = converted_df["ID"].map(data["lineage"])
    converted_df.to_csv("./data/converted_data.csv", index=False)

    return converted_df, target_length


def train(model, criterion, optimizer, train_loader, device, num_epochs, num_classes):
    training_loss = []  # To record training loss after every epoch

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.float().to(device)
            inputs = inputs.unsqueeze(1)  # Add an extra channel dimension

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
            training_loss.append(running_loss / len(train_loader))
    return training_loss


def evaluate(model, test_loader, device):
    model.eval()
    total = 0
    correct = 0
    top5_correct = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc="Evaluating"):
            sequences, labels = sequences.float().to(device), labels.to(device)
            sequences = sequences.unsqueeze(1)  # Add an extra channel dimension
            outputs = model(sequences).squeeze(1)  # squeeze the 2nd dimension
            _, predicted = torch.max(outputs, 1)
            _, top5_pred = torch.topk(outputs, 5)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            top5_correct += sum([1 if labels[i] in top5_pred[i] else 0 for i in range(len(labels))])

    top1_accuracy = correct / total
    top5_accuracy = top5_correct / total
    print(f"Top 1 Accuracy: {top1_accuracy*100:.2f}%")
    print(f"Top 5 Accuracy: {top5_accuracy*100:.2f}%")

    return torch.tensor(all_predictions), torch.tensor(all_labels), top1_accuracy, top5_accuracy


def visualize_results(test_labels, predicted_onehot, predicted, num_classes):
    test_labels_bin = label_binarize(test_labels.cpu().numpy(), classes=[i for i in range(num_classes)])
    predicted_onehot = predicted_onehot.cpu().numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], predicted_onehot[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall curve
    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(test_labels_bin[:, i], predicted_onehot[:, i])
        average_precision[i] = average_precision_score(test_labels_bin[:, i], predicted_onehot[:, i])

    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    for i, color in zip(range(num_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-recall curve of class {0} (area = {1:0.2f})'
                       ''.format(i, average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(test_labels.cpu().numpy(), predicted.cpu().numpy())
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
    plt.ylabel('Actual label');
    plt.xlabel('Predicted label');
    plt.show()

    # Classification Report
    print(classification_report(test_labels.cpu().numpy(), predicted.cpu().numpy()))


def main():
    # convdf,  input_size = load_data()
    data = pd.read_csv("./data/dataset.csv")
    sequence_lengths = data["sequence_length"].values
    input_size = sequence_lengths.mean().astype(int)
    convdf = pd.read_csv("./data/converted_data.csv")

    convdf['lineage'] = convdf['lineage'].astype('category')

    sequences = convdf['Enc_Sequence']
    labels = convdf['lineage']
    # for seq in sequences:
    #     enc_sequence = np.array(list(seq), dtype=np.uint8)
    #     enc_sequence = enc_sequence.reshape(-1)  # Remove extra dimensions
    #     seq = torch.tensor(enc_sequence)


    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)


    X_train_list = X_train.apply(lambda x: [int(num) for num in x]).tolist()
    X_train_tensor = torch.tensor(X_train_list)
    y_train_tensor = torch.tensor(y_train.cat.codes.tolist())

    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_data, batch_size=32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = len(labels.unique())
    hidden_size = 32
    model = ComplexCNN(input_size, hidden_size, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model_name = type(model).__name__

    if os.path.exists(f"./model/{model_name}.pth"):
        model.load_state_dict(torch.load(f"./model/{model_name}.pth"))
    else:
        training_loss = train(model, criterion, optimizer, train_loader, device, num_epochs=30, num_classes=num_classes)
        torch.save(model.state_dict(), f"./model/{model_name}.pth")

    X_test_list = X_test.apply(lambda x: [int(num) for num in x]).tolist()
    X_test_tensor = torch.tensor(X_test_list)
    y_test_tensor = torch.tensor(y_test.cat.codes.tolist())

    test_data = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_data, batch_size=32)
    predicted, test_labels, top1_accuracy, top5_accuracy = evaluate(model, test_loader, device)
    predicted = predicted.long()
    predicted_onehot = torch.nn.functional.one_hot(predicted, num_classes=num_classes)
    visualize_results(test_labels, predicted_onehot, predicted, num_classes)

    # Plot Learning Curve
    plt.figure()
    plt.plot(training_loss)
    plt.title('Learning Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()
