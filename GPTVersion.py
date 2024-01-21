import pandas as pd
import numpy as np
import tensorflow as tf
import evaluate
from transformers import DataCollatorWithPadding, create_optimizer, GPT2Tokenizer, TFGPT2ForSequenceClassification, \
    AutoTokenizer, GPT2Config
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# set GPU memory, can be ignored
using_gpu_index = 0
gpu_list = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_list) > 0:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpu_list[using_gpu_index],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10*1024)]  # limit the size of GPU memory
        )
    except RuntimeError as e:
        print(e)
else:
    print("Got no GPUs")

# read data
train_df = pd.read_csv("./dataset/preprocessed_train.csv")
x, y = train_df['content'].to_numpy(), train_df['category'].to_numpy()

# split data
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, stratify=y)
ds_train = Dataset.from_dict({"text": train_x, "label": train_y})
ds_valid = Dataset.from_dict({"text": valid_x, "label": valid_y})
ds = DatasetDict()
ds["train"] = ds_train
ds["valid"] = ds_valid


# tokenize
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = '[PAD]'
tokenized_ds = ds.map(lambda examples: tokenizer(examples["text"],padding=True,truncation=True), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# conversion between id and label
category_df = pd.read_csv("dataset/category_dict.csv")
id2label = {}
label2id = {}
for i in range(len(category_df)):
    category_id = int(category_df["category_id"].loc[i])
    category_name = category_df["category_name"].loc[i]
    id2label[category_id] = category_name
    label2id[category_name] = category_id

# evaluation
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# model construction
batch_size = 2
num_epochs = 5
batches_per_epoch = len(ds_train) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)
model = TFGPT2ForSequenceClassification(GPT2Config()).from_pretrained("gpt2", num_labels=32, id2label=id2label, label2id=label2id)
model.config.pad_token_id = tokenizer.pad_token_id
model.load_weights('NC1/NCG.weights.h5')
model.compile(optimizer=optimizer)

# train
tf_train_set = model.prepare_tf_dataset(
    tokenized_ds['train'],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    tokenized_ds['valid'],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)
metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
model_checkpoint = ModelCheckpoint("NC1/NCG.weights.h5", save_best_only=True,save_weights_only=True)
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=1, callbacks=[metric_callback,model_checkpoint])

# accuracy 0.7412