from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, Flatten
from keras.layers import Dense, Dropout
from keras import Input


class CNN1D(Model):
    def __init__(self, sequence_length, n_classes, dropout_rate=0.3):
        super(CNN1D, self).__init__()

        self.conv1 = Conv1D(filters=32, kernel_size=5, activation='relu')
        self.pool1 = MaxPooling1D(pool_size=2)
        self.conv2 = Conv1D(filters=64, kernel_size=5, activation='relu')
        self.pool2 = MaxPooling1D(pool_size=2)
        self.flatten = Flatten()  # Flatten layer
        self.fc1 = Dense(64, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Dense(n_classes, activation='softmax')

    def call(self, x):
        print(f"Initial input shape: {x.shape}")

        x = self.conv1(x)
        print(f"After Conv1D-1 output shape: {x.shape}")

        x = self.pool1(x)
        print(f"After MaxPooling1D-1 output shape: {x.shape}")

        x = self.conv2(x)
        print(f"After Conv1D-2 output shape: {x.shape}")

        x = self.pool2(x)
        print(f"After MaxPooling1D-2 output shape: {x.shape}")

        x = self.flatten(x)
        print(f"After Flatten output shape: {x.shape}")  # Apply flatten layer here

        x = self.fc1(x)
        print(f"After Dense-1 output shape: {x.shape}")

        x = self.dropout(x)
        print(f"After Dropout output shape: {x.shape}")

        x = self.fc2(x)
        print(f"Final output shape: {x.shape}")

        return x
