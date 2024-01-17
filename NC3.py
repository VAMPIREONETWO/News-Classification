from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from NCModel import NCModel3
import pandas as pd

# set GPU memory, can be ignored
using_gpu_index = 0
gpu_list = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_list) > 0:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpu_list[using_gpu_index],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)]  # limit the size of GPU memory
        )
    except RuntimeError as e:
        print(e)
else:
    print("Got no GPUs")

# parameters
vocab_size = 7000
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# train-test split
train_df = pd.read_csv("./dataset/preprocessed_train.csv")
x, y = train_df['content'].to_numpy(), train_df['category'].to_numpy().reshape(len(train_df['category']), 1)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, stratify=y, shuffle=True)
# tokenize sentences
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_x)

# convert train dataset to sequence and pad sequences
train_x = tokenizer.texts_to_sequences(train_x)
train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, padding=padding_type, truncating=trunc_type,
                                                        maxlen=max_length)
# convert valid dataset to sequence and pad sequences
valid_x = tokenizer.texts_to_sequences(valid_x)
valid_x = tf.keras.preprocessing.sequence.pad_sequences(valid_x, padding=padding_type, truncating=trunc_type,
                                                        maxlen=max_length)
# model construction
model = NCModel3(vocab_size=vocab_size, max_length=max_length, classes=32)
model.build((None, 200))
model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer=Adam(),
              metrics=['accuracy'])
model.summary()

# train and evaluate
history = model.fit(train_x, train_y, batch_size=32, epochs=10)
model.evaluate(valid_x, valid_y, return_dict=True)

# accuracy: 0.6463
