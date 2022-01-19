import random
import os
import sys
import keras
from wordpiece import CharTokenizer
import tensorflow as tf
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.get_logger().setLevel('ERROR')

import numpy as np

def load_data(file_path):
    srcs = []
    tgts = []
    with open(file_path) as fp:
        for line in fp:
            src, tgt = line.strip().split("\t")
            srcs.append(src)
            tgts.append(tgt)
    return srcs, tgts

def format_dataset(src, tgt):
    return ({"encoder_inputs": src, "decoder_inputs": tgt[:, :-1],}, tgt[:, 1:])

def make_dataset(src, tgt, batch_size):
    src = np.asarray(src)
    tgt = np.asarray(tgt)
    dataset = tf.data.Dataset.from_tensor_slices((src, tgt))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache()

def coshuffle(x, y):
    pairs = list(zip(x, y))
    random.shuffle(pairs)
    x, y = zip(*pairs)
    return list(x), list(y)

files = ["datasets/transliterate/en_hi.txt"]
src, tgt = [], []

maxlen = 20
src_vectorizer = CharTokenizer(maxlen)
tgt_vectorizer = CharTokenizer(maxlen + 1)

vocab_size = src_vectorizer.vocab_size

for f in files:
    x, y = load_data(f)
    src += x
    tgt += y

src = [src_vectorizer.tokenize(token) for token in src]
tgt = [tgt_vectorizer.tokenize(token) for token in tgt]


src, tgt = coshuffle(src,  tgt)
for i in random.sample(list(zip(src, tgt)), 10):
    s, t = i
    s = src_vectorizer.detokenize(s)
    t = tgt_vectorizer.detokenize(t)
    print(s, t)


split_point = int(len(src) * .85)
train_src, val_src = src[:split_point], src[split_point:]
train_tgt, val_tgt = tgt[:split_point], tgt[split_point:]

train_ds = make_dataset(train_src, train_tgt, 64)
val_ds = make_dataset(val_src, val_tgt, 64)

for elem in tqdm(train_ds):
    pass

for elem in tqdm(val_ds):
    pass

from modellib import Seq2SeqTransformer
transformer = Seq2SeqTransformer.get(maxlen, vocab_size, tgt_vocab_size=vocab_size)
transformer.summary()
transformer.compile(
    "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

epochs = 2
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)
transformer.save("checkpoints/transliterator_transformer.hdf5")
