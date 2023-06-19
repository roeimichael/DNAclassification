import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
    # Load labels
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
    # pad sequence if its length is less than the desired sequence_length
    if len(sequence) < sequence_length:
        sequence = np.pad(sequence, pad_width=(0, sequence_length - len(sequence)), mode='constant')
    sequence = sequence.reshape(-1, 1)
    return sequence


def load_and_preprocess_data(file, label, sequence_length):
    path = tf.strings.join(["./data/encoded_sequences/", file])
    encoded_sequence = tf.numpy_function(numpy_load, [path, sequence_length], tf.float32)
    encoded_sequence.set_shape([None, 1])  # Set the shape manually as tf.numpy_function does not set it
    return encoded_sequence, label


def create_datasets(files_train, y_train, files_test, y_test, sequence_length):
    # Create a tf.data.Dataset object
    train_dataset = tf.data.Dataset.from_tensor_slices((files_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((files_test, y_test))

    # Apply the load_and_preprocess_data function
    train_dataset = train_dataset.map(lambda x, y: load_and_preprocess_data(x, y, sequence_length))
    test_dataset = test_dataset.map(lambda x, y: load_and_preprocess_data(x, y, sequence_length))

    # Shuffle and batch the datasets
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(32)
    test_dataset = test_dataset.shuffle(buffer_size=1000).batch(32)

    return train_dataset, test_dataset


def create_model(sequence_length, num_classes):
    # Define the model
    model = Sequential()
    model.add(Conv1D(64, 5, activation='relu', input_shape=(sequence_length, 1)))  # Convolutional layer
    model.add(Dropout(0.5))  # Dropout layer to reduce overfitting
    model.add(Flatten())  # Flatten layer
    model.add(Dense(128, activation='relu'))  # Dense layer with 128 neurons
    model.add(Dense(num_classes, activation='softmax'))  # Output layer
    model.summary()
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[CategoricalAccuracy()])

    return model


def evaluate_model(y_test, y_pred):
    # Convert one-hot encoded labels back to class labels
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    return accuracy, precision, recall, f1


def save_results(model, accuracy, precision, recall, f1):
    # Store results
    results = pd.DataFrame({
        'model_name': ['CNN'],
        'parameters': [str(model.get_config())],
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'F1_score': [f1]
    })
    results.to_csv("model_evaluation.csv", index=False)


# Main function to orchestrate all the steps
def main():
    check_gpu_memory()

    # Load and preprocess data
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
    model.summary()
    model.fit(train_dataset, validation_data=test_dataset, epochs=10)

    # Predict on test data
    y_pred = model.predict(test_dataset)

    # Evaluate model
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

    # Save results
    save_results(model, accuracy, precision, recall, f1)


if __name__ == "__main__":
    main()
