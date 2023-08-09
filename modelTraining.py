import logging
import pandas as pd
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from models.ComplexCNN import ComplexCNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, \
    average_precision_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


class GenomicDataset(Dataset):
    def __init__(self, file_paths, labels, target_length):
        self.file_paths = file_paths
        self.labels = labels
        self.target_length = target_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], "r") as file:
            content = file.read().strip()
            difference = self.target_length - len(content)
            if difference > 0:
                content += "0" * difference
            elif difference < 0:
                content = content[:self.target_length]
            content = [int(c) for c in content]  # Convert string of '0's and '1's to a list of integers
            return torch.tensor(content, dtype=torch.long), self.labels[idx]


class GenomicClassifier:
    def __init__(self, model, criterion, optimizer, scheduler, device, num_epochs, num_classes):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.training_loss = []
        self.target_length = 0

    def load_data(self, num_lineages=200, samples_per_lineage=50):
        """
        Load genomic data from disk and prepare data loaders.
        """
        text_files_folder = "./data/encoded_sequences"
        assert os.path.exists(text_files_folder), "Data folder does not exist."

        all_lineages = [d for d in os.listdir(text_files_folder) if os.path.isdir(os.path.join(text_files_folder, d))]
        selected_lineages = np.random.choice(all_lineages, num_lineages, replace=False)
        label_encoder = LabelEncoder()
        encoded_lineages = label_encoder.fit_transform(selected_lineages)
        class_to_lineage = {class_id: lineage for lineage, class_id in zip(selected_lineages, encoded_lineages)}

        converted_file_paths = []
        converted_labels = []
        sequence_lengths = []

        for lineage, class_id in tqdm(zip(selected_lineages, encoded_lineages), desc="Processing lineages"):
            lineage_folder = os.path.join(text_files_folder, lineage)
            files = os.listdir(lineage_folder)[:samples_per_lineage]
            for file_path in files:
                full_file_path = os.path.join(lineage_folder, file_path)
                with open(full_file_path, "r") as file:
                    content = file.read().strip()
                    sequence_lengths.append(len(content))
                    converted_file_paths.append(full_file_path)
                    converted_labels.append(class_id)

        self.target_length = int(np.mean(sequence_lengths))
        train_file_paths, test_file_paths, y_train, y_test = train_test_split(converted_file_paths, converted_labels,
                                                                              test_size=0.2, random_state=42)
        train_data = GenomicDataset(train_file_paths, y_train, self.target_length)
        test_data = GenomicDataset(test_file_paths, y_test, self.target_length)
        return DataLoader(train_data, batch_size=8), DataLoader(test_data, batch_size=8), class_to_lineage

    def train(self, train_loader):
        """
        Train the model using the provided data loader.
        """
        self.training_loss = []
        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device).float().unsqueeze(1)  # Add channel dimension
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == targets).sum().item()
                if batch_idx % 10 == 9:
                    batch_loss = running_loss / 10
                    accuracy = 100 * correct_predictions / (10 * len(inputs))
                    logging.info(
                        f"Epoch [{epoch + 1}/{self.num_epochs}] - Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {batch_loss:.4f} - Accuracy: {accuracy:.2f}%")
                    running_loss = 0.0
                    correct_predictions = 0
                self.training_loss.append(running_loss / len(train_loader))
        return self.training_loss

    def evaluate(self, test_loader, class_to_lineage):
        self.model.eval()
        lineage_metrics = {lineage: {'total': 0, 'correct': 0, 'top3_correct': 0, 'top5_correct': 0} for lineage in
                           class_to_lineage.values()}

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for sequences, labels in tqdm(test_loader, desc="Evaluating"):
                sequences, labels = sequences.float().to(self.device), labels.to(self.device)
                sequences = sequences.unsqueeze(1)  # Add an extra channel dimension
                outputs = self.model(sequences).squeeze(1)  # squeeze the 2nd dimension

                _, predicted = torch.max(outputs, 1)
                _, top3_pred = torch.topk(outputs, 3)
                _, top5_pred = torch.topk(outputs, 5)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                for i in range(len(labels)):
                    lineage_name = class_to_lineage[labels[i].item()]
                    lineage_metrics[lineage_name]['total'] += 1
                    lineage_metrics[lineage_name]['correct'] += (predicted[i] == labels[i]).item()
                    lineage_metrics[lineage_name]['top3_correct'] += (labels[i] in top3_pred[i]).item()
                    lineage_metrics[lineage_name]['top5_correct'] += (labels[i] in top5_pred[i]).item()

        metrics_data = []
        for lineage, metrics in lineage_metrics.items():
            if metrics['total'] > 0:
                accuracy = metrics['correct'] / metrics['total']
                precision = precision_score(all_labels, all_predictions, labels=[class_to_lineage[lineage]],
                                            average='micro')
                top3acc = metrics['top3_correct'] / metrics['total']
                top5acc = metrics['top5_correct'] / metrics['total']
                metrics_data.append([lineage, accuracy, precision, top3acc, top5acc])

        results_df = pd.DataFrame(metrics_data, columns=['Lineage', 'accuracy', 'precision', 'top3acc', 'top5acc'])
        results_df.to_csv("results.csv", index=False)


def main():
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {device}")

        classifier = GenomicClassifier(model=None, criterion=None, optimizer=None, scheduler=None, device=device,
                                       num_epochs=50, num_classes=None)
        train_loader, test_loader, class_to_lineage = classifier.load_data(num_lineages=20, samples_per_lineage=250)

        num_classes = len(class_to_lineage)
        target_length = classifier.target_length  # Get the target length
        model = ComplexCNN(input_size=target_length, hidden_size=16, num_classes=num_classes)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        classifier.model = model
        classifier.criterion = criterion
        classifier.optimizer = optimizer
        classifier.scheduler = scheduler
        classifier.num_classes = num_classes
        classifier.train(train_loader)
        classifier.evaluate(test_loader, class_to_lineage)
        # Results visualization - Using training loss as an example
        plt.plot(classifier.training_loss)
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
