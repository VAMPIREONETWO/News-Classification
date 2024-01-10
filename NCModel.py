import tensorflow as tf
import keras
from keras.layers import Conv1D, Dense, Embedding, GlobalAveragePooling1D, Activation, BatchNormalization


class NCModel(keras.Model):
    def __init__(self, input_dim, input_length, classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eb = Embedding(input_dim=input_dim, output_dim=64,
                            input_shape=(None,input_length), input_length=input_length)

        self.cl1 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same")
        self.bn1 = BatchNormalization()
        self.relu1 = Activation("relu")
        self.cl2 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same")
        self.bn2 = BatchNormalization()
        self.relu2 = Activation("relu")
        self.cl3 = Conv1D(filters=128, kernel_size=3, strides=1, padding="same")
        self.bn3 = BatchNormalization()
        self.relu3 = Activation("relu")
        self.pool1 = GlobalAveragePooling1D()

        self.fc = Dense(classes, activation='softmax')

    def call(self, inputs, training=None, mask=None):
        outputs = self.eb(inputs)
        outputs = self.cl1(outputs)
        outputs = self.bn1(outputs)
        outputs = self.relu1(outputs)
        outputs = self.cl2(outputs)
        outputs = self.bn2(outputs)
        outputs = self.relu2(outputs)
        outputs = self.cl3(outputs)
        outputs = self.bn3(outputs)
        outputs = self.relu3(outputs)
        outputs = self.pool1(outputs)
        outputs = self.fc(outputs)
        return outputs
