from keras.layers import Conv2D, MaxPooling2D ,Flatten
from keras.models import Model
from keras.layers import Dense, Dropout

class CNN2D(Model):
    def __init__(self, input_shape, n_classes, dropout_rate=0.3):
        super(CNN2D, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(4, 4), activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(1, 2))
        self.conv2 = Conv2D(filters=64, kernel_size=(4, 4), activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(1, 2))
        self.flatten = Flatten()
        self.fc1 = Dense(64, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Dense(n_classes, activation='softmax')

    def call(self, x):
        print(f"Initial input shape: {x.shape}")
        x = self.conv1(x)
        print(f"After Conv2D-1 output shape: {x.shape}")
        x = self.pool1(x)
        print(f"After MaxPooling2D-1 output shape: {x.shape}")
        x = self.conv2(x)
        print(f"After Conv2D-2 output shape: {x.shape}")
        x = self.pool2(x)
        print(f"After MaxPooling2D-2 output shape: {x.shape}")
        x = self.flatten(x)
        print(f"After Flatten output shape: {x.shape}")
        x = self.fc1(x)
        print(f"After Dense-1 output shape: {x.shape}")
        x = self.dropout(x)
        print(f"After Dropout output shape: {x.shape}")
        x = self.fc2(x)
        print(f"Final output shape: {x.shape}")

        return x
