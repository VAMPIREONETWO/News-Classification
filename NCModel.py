import tensorflow as tf
import keras
from keras.layers import Conv1D, Dense, Embedding, GlobalAveragePooling1D, Activation, BatchNormalization, Layer, \
    Dropout, LSTM, TextVectorization, Input


class NCModel(keras.Model):
    def __init__(self, vocab_size, max_length,vocab,classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # preprocessing layers
        tv = TextVectorization(max_tokens=vocab_size, output_mode='int', output_sequence_length=max_length)
        tv.adapt(vocab)
        self.pp = keras.Sequential(
            layers=[Input(shape=(1,), dtype=tf.string),
                    tv,
                    Embedding(input_dim=vocab_size,output_dim=64,input_length=max_length)]
        )
        self.lstm = LSTM(32)
        self.fc = Dense(classes,activation="softmax")

    def call(self, inputs, training=None, mask=None):
        outputs = self.pp(inputs)
        outputs = self.lstm(outputs)
        outputs = self.fc(outputs)
        return outputs


class CNNLayer(Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.cl = Conv1D(filters=filters, kernel_size=2, strides=1, padding="same")
        self.bn = BatchNormalization()
        self.relu = Activation("relu")

    def call(self, inputs, *args, **kwargs):
        outputs = self.cl(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
        return outputs
