import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
from keras.optimizers import Adam
from keras.metrics import CategoricalAccuracy, Precision, Recall
from sklearn.metrics import classification_report
from keras.utils import plot_model
from CNN1D import CNN1D

warnings.filterwarnings("ignore")


def check_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def load_data():
    # Load the list of encoded sequence file names
    encoded_files = os.listdir("./data/encoded_sequences/")
    reduced_data = pd.read_csv("./data/dataset.csv")
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(reduced_data["lineage"])
    return encoded_files, labels, label_encoder


def split_data(encoded_files, labels):
    # Create a train-test split
    files_train, files_test, y_train, y_test = train_test_split(
        encoded_files,
        labels,
        test_size=0.2,
        random_state=42
    )

    # Convert labels to one-hot
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return files_train, files_test, y_train, y_test


def numpy_load(file, sequence_length):
    sequence = np.load(file.decode('utf-8')).astype(np.float32)
    print(f"Original sequence shape: {sequence.shape}")
    if len(sequence) < sequence_length:
        sequence = np.pad(sequence, pad_width=(0, sequence_length - len(sequence)), mode='constant')
    print(f"Padded sequence shape: {sequence.shape}")
    sequence = sequence.reshape(-1, 1)  # Note that we reshaped to (-1, 1) here
    sequence = np.expand_dims(sequence, axis=0)  # Add another dimension to get a 2D array. The shape becomes (1, sequence_length)
    print(f"Reshaped sequence shape: {sequence.shape}")
    return sequence


def create_datasets(files_train, y_train, files_test, y_test, sequence_length):
    def load_and_preprocess_data(file, label, sequence_length):
        sequence = np.load(file).astype(np.float32)
        # print(f"Original sequence shape: {sequence.shape}")
        if len(sequence) < sequence_length:
            sequence = np.pad(sequence, pad_width=(0, sequence_length - len(sequence)), mode='constant')
        # print(f"Padded sequence shape: {sequence.shape}")
        sequence = sequence.reshape(-1, 1)  # Note that we reshaped to (-1, 1) here
        # print(f"Reshaped sequence shape: {sequence.shape}")
        sequence = tf.convert_to_tensor(sequence)
        label = tf.convert_to_tensor(label)
        return sequence, label

    train_sequences, train_labels, test_sequences, test_labels = [], [], [], []
    for file, label in zip(files_train, y_train):
        sequence, label = load_and_preprocess_data("./data/encoded_sequences/" + file, label, sequence_length)
        train_sequences.append(sequence)
        train_labels.append(label)

    for file, label in zip(files_test, y_test):
        sequence, label = load_and_preprocess_data("./data/encoded_sequences/" + file, label, sequence_length)
        test_sequences.append(sequence)
        test_labels.append(label)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_sequences, train_labels)).shuffle(buffer_size=1000).batch(16)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_sequences, test_labels)).shuffle(buffer_size=1000).batch(16)

    return train_dataset, test_dataset


def create_model(sequence_length, n_classes):
    model = CNN1D(sequence_length, dropout_rate=0.5,n_classes=n_classes)
    model.compile(
        loss=CategoricalCrossentropy(),
        optimizer=Adam(),
        metrics=[CategoricalAccuracy(), Precision(), Recall()]
    )
    return model


def evaluate_model(y_true, y_pred):
    # Convert prediction probabilities to class labels
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    # Compute metrics
    accuracy = np.mean(y_true == y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']

    return accuracy, precision, recall, f1


def save_results(model, accuracy, precision, recall, f1):
    model.save('model.h5')
    print(f'Saved model to disk.')

    print('\nClassification Report:\n')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    # Optionally, save the model architecture as an image
    plot_model(model, to_file='model.png', show_shapes=True)
    print('Saved model architecture to disk.')


def main():
    check_gpu_memory()
    encoded_files, labels, label_encoder = load_data()
    files_train, files_test, y_train, y_test = split_data(encoded_files, labels)

    # Load one file to get the sequence length
    sample_sequence = np.load(f"./data/encoded_sequences/{encoded_files[0]}")
    sequence_length = sample_sequence.shape[0]
    print("Sequence Length:", sequence_length)

    # Create datasets
    train_dataset, test_dataset = create_datasets(files_train, y_train, files_test, y_test, sequence_length)

    # Create and train model
    model = create_model(sequence_length, len(label_encoder.classes_))
    model.fit(train_dataset, validation_data=test_dataset, epochs=10)

    # Predict on test data
    y_pred = model.predict(test_dataset)

    # As the data is in batches, we need to unbatch it before evaluation
    y_test = np.concatenate([y for x, y in test_dataset], axis=0)

    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')
    save_results(model, accuracy, precision, recall, f1)


if __name__ == "__main__":
    main()