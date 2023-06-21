import pickle

import tensorflow as tf
import warnings
from CNN2D import CNN2D
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.metrics import CategoricalAccuracy, Precision, Recall
from keras.utils import plot_model

warnings.filterwarnings("ignore")


def check_gpu_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def create_model_2D(sequence_length, n_classes):
    model = CNN2D(sequence_length, n_classes, dropout_rate=0.5)
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
    plot_model(model, to_file='model.png', show_shapes=True)
    print('Saved model architecture to disk.')


def load_data():
    sequences = []
    labels = []
    reduced_data = pd.read_csv("./data/reduced_data_encoded.csv")
    reduced_data["id"] = reduced_data["id"].str.strip('"')
    with open("./data/label_encoder.pkl", "rb") as file:
        label_encoder = pickle.load(file)
    npy_files = os.listdir("./data/encoded_sequences/")
    for npy_file in npy_files:
        id = npy_file.split(".")[0]
        sequence = np.load(f"./data/encoded_sequences/{npy_file}")
        lineage_values = reduced_data.loc[reduced_data["id"] == id, "lineage"].values
        if len(lineage_values) > 0:
            label = label_encoder.transform([lineage_values[0]])[0]
        else:
            print(f"No match found for id {id}")
            label = -1
        sequences.append(sequence)
        labels.append(label)

    sequences = np.array(sequences)
    sequences = sequences.reshape((-1, 3000, 4, 1))

    labels = to_categorical(np.array(labels))  # Convert labels to one-hot encoded format
    return sequences, labels


sequences, labels = load_data()
print(len(sequences), len(labels))

X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Verify the length of the training and testing sets
print(len(X_train), len(X_test), len(y_train), len(y_test))
input_shape = (X_train.shape[1], 4, 1)  # Adjusted input shape
n_classes = y_train.shape[1]

model = CNN2D(input_shape, n_classes, dropout_rate=0.5)
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(),
    metrics=[CategoricalAccuracy(), Precision(), Recall()]
)

# Fit the model to the training data
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)

# Save the model and the evaluation results
save_results(model, accuracy, precision, recall, f1)
