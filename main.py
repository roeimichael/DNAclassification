import os
import pandas as pd
from keras.preprocessing import text, sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm


# Function to convert sequence strings into binary matrix
def sequences_to_matrix(seq):
    tokenizer = text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(seq)
    binary_matrix = tokenizer.texts_to_matrix(seq)
    return binary_matrix

# Function to read fasta files
def read_fasta_file(path):
    with open(path, "r") as file:
        return file.read().split("\n", 1)[1].replace("\n", "")

# Load the reduced metadata
reduced_data = pd.read_csv("./data/reduced_data.csv")

# Read each fasta file, extract the sequence, and find the corresponding label
fasta_files = os.listdir("./data/fasta_files/")
for fasta_file in tqdm(fasta_files, desc="Processing fasta files"):
    id = fasta_file.split("_")[1]
    sequence = read_fasta_file(f"./data/fasta_files/{fasta_file}")
    reduced_data.loc[reduced_data["id"] == id, "sequence"] = sequence

reduced_data.to_csv("./data/reduced_data.csv")
exit()
# Convert the sequences to binary matrix format
reduced_data["sequence"] = reduced_data["sequence"].apply(sequences_to_matrix)


# Convert the labels to one-hot encoding
labels = to_categorical(reduced_data["label"].astype('category').cat.codes)

# Split the data into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(reduced_data["sequence"].values, labels, test_size=0.2)

# Define the model
model = Sequential()
model.add(Conv1D(64, 5, activation='relu', input_shape=(x_train.shape[1], 1)))
model.add(Flatten())
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[categorical_accuracy])

# Fit the model to the training data
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
# Predict the test set results
y_pred = model.predict(x_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Create a dataframe for the results
results = pd.DataFrame({
    'model_name': ['CNN'],
    'parameters': [str(model.get_config())],
    'accuracy': [accuracy],
    'precision': [precision],
    'recall': [recall],
    'F1_score': [f1]
})

# Save the results to a CSV file
results.to_csv("model_evaluation.csv", index=False)