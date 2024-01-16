from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from NCModel import NCModel2
import re
import pandas as pd

# Stopwords list from https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js
stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]


# removing non alphanumeric character
def alpha_num(text):
    return re.sub(r'[^A-Za-z0-9 ]', '', text)


# removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stopwords:
            final_text.append(i.strip())
    return " ".join(final_text)


def preprocess(df):
    df['title'] = df['title'].str.lower()
    df['title'] = df['title'].apply(alpha_num)
    df['title'] = df['title'].apply(remove_stopwords)
    return df


train_df = pd.read_csv("./dataset/preprocessed_train.csv")
# parameters
vocab_size = 7000
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
x, y = train_df['content'].to_numpy(), train_df['category'].to_numpy().reshape(len(train_df['category']), 1)
# train-test split
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
model = NCModel2(vocab_size=vocab_size, max_length=max_length,classes=32)
model.build((None,200))
model.compile(loss=SparseCategoricalCrossentropy(),
              optimizer=Adam(),
              metrics=['accuracy'])
model.summary()

# train and evaluate
history = model.fit(train_x, train_y, batch_size=32, epochs=10)
model.evaluate(valid_x, valid_y, return_dict=True)

# accuracy: 0.6445
