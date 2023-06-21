from keras.models import Model
from keras.layers import Dense, Dropout


class MLP(Model):
    def __init__(self, input_dim, n_classes, dropout_rate=0.3):
        super(MLP, self).__init__()

        self.fc1 = Dense(64, activation='relu')
        self.dropout = Dropout(dropout_rate)
        self.fc2 = Dense(32, activation='relu')
        self.fc3 = Dense(n_classes, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
