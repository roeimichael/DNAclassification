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
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, \
    average_precision_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


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

    def load_data(self):
        try:
            text_files_folder = "./data/encoded_sequences"
            assert os.path.exists(text_files_folder), "Data folder does not exist."

            lineages = [d for d in os.listdir(text_files_folder) if
                        os.path.isdir(os.path.join(text_files_folder, d))]
            label_encoder = LabelEncoder()
            encoded_lineages = label_encoder.fit_transform(lineages)
            class_to_lineage = {class_id: lineage for lineage, class_id in zip(lineages, encoded_lineages)}
            converted_data = []
            sequence_lengths = []

            # Wrap the outer loop with tqdm
            for lineage, class_id in tqdm(zip(lineages, encoded_lineages), desc="Processing lineages"):
                lineage_folder = os.path.join(text_files_folder, lineage)
                files = os.listdir(lineage_folder)[:100]

                # Optionally, wrap the inner loop with tqdm
                for file_path in files:
                    with open(os.path.join(lineage_folder, file_path), "r") as file:
                        content = file.read().strip()
                        sequence_lengths.append(len(content))
                        converted_data.append([file_path, "".join(map(str, content)), class_id])

            target_length = int(np.mean(sequence_lengths))

            for row in tqdm(converted_data, desc="Converting data"):
                difference = target_length - len(row[1])
                if difference > 0:
                    padding = "0" * difference
                    row[1] += padding
                elif difference < 0:
                    row[1] = row[1][:target_length]

            converted_df = pd.DataFrame(converted_data, columns=["ID", "Enc_Sequence", "lineage"])
            converted_df.to_csv("./data/converted_data.csv", index=False)
            return converted_df, target_length, class_to_lineage
        except Exception as e:
            logging.error(f"An error occurred during data loading: {str(e)}")
            raise

    def train(self, train_loader):
        self.training_loss = []  # To record training loss after every epoch

        for epoch in tqdm(range(self.num_epochs), desc="Training"):
            # Existing code
            self.model.train()
            running_loss = 0.0
            correct_predictions = 0

            for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader),desc=f"Epoch {epoch + 1}/{self.num_epochs}"):
                inputs = inputs.float().to(self.device)
                inputs = inputs.unsqueeze(1)  # Add an extra channel dimension

                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)  # Pass inputs to the model

                targets = targets.view(-1)
                outputs = outputs.view(-1, self.num_classes)

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
                    print(
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

        return results_df

    def visualize_results(self, test_labels, predicted_onehot, predicted):
        test_labels_bin = label_binarize(test_labels.cpu().numpy(), classes=[i for i in range(self.num_classes)])
        predicted_onehot = predicted_onehot.cpu().numpy()

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], predicted_onehot[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

        for i, color in zip(range(self.num_classes), colors):
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

        for i in range(self.num_classes):
            precision[i], recall[i], _ = precision_recall_curve(test_labels_bin[:, i], predicted_onehot[:, i])
            average_precision[i] = average_precision_score(test_labels_bin[:, i], predicted_onehot[:, i])

        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        for i, color in zip(range(self.num_classes), colors):
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
    try:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Device: {device}")

        classifier = GenomicClassifier(model=None, criterion=None, optimizer=None, scheduler=None, device=device,
                                       num_epochs=30, num_classes=None)
        convdf, input_size, class_to_lineage = classifier.load_data()

        convdf['lineage'] = convdf['lineage'].astype('category').cat.codes
        sequences = convdf['Enc_Sequence']
        labels = convdf['lineage']
        X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

        X_train_tensor = torch.tensor(X_train.apply(lambda x: [int(num) for num in x]).tolist())
        y_train_tensor = torch.tensor(y_train.tolist())
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_data, batch_size=16)

        num_classes = len(labels.unique())
        hidden_size = 32
        model = ComplexCNN(input_size, hidden_size, num_classes)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        classifier.model = model
        classifier.criterion = criterion
        classifier.optimizer = optimizer
        classifier.scheduler = scheduler
        classifier.num_classes = num_classes

        classifier.training_loss = classifier.train(train_loader)

        X_test_tensor = torch.tensor(X_test.apply(lambda x: [int(num) for num in x]).tolist())
        y_test_tensor = torch.tensor(y_test.tolist())
        test_data = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_data, batch_size=16)

        results_df = classifier.evaluate(test_loader, class_to_lineage)

        print(results_df)

        plt.figure()
        plt.plot(classifier.training_loss)
        plt.title('Learning Curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()
    except Exception as e:
        logging.error(f"An error occurred in the main function: {str(e)}")
        raise


if __name__ == "__main__":
    main()
