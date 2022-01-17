import random
import os
import numpy as np
import tensorflow as tf

from wordpiece import SubWordTokenizer
from tqdm import tqdm
import tensorflow_text

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)



def load_data(file_path):
    srcs = []
    tgts = []
    with open(file_path) as fp:
        for line in fp:
            src, tgt = line.strip()
            srcs.append(src)
            tgts.append(tgt)
    return srcs, tgts

def format_dataset(src, tgt):
    return ({"encoder_inputs": src, "decoder_inputs": tgt[:, :-1],}, tgt[:, 1:])

def make_dataset(src, tgt, batch_size):
    dataset = tf.data.Dataset.zip((src, tgt))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    dataset = dataset.prefetch(128).cache(filename="datasets_serialized.tfrecord").shuffle(128)
    return dataset

maxlen = 25
en_vectorizer = SubWordTokenizer(maxlen=maxlen)
en_vectorizer.load("en_new_vocab.txt")

hi_vectorizer = SubWordTokenizer(maxlen=maxlen + 1)
hi_vectorizer.load("hi_new_vocab.txt")

src_file = "datasets/parallel/samanantar.en"
tgt_file = "datasets/parallel/samanantar.hi"

# src_file = "datasets/parallel/IITB.en-hi.en"
# tgt_file = "datasets/parallel/IITB.en-hi.hi"

src = tf.data.TextLineDataset(src_file).map(en_vectorizer.tokenize)
tgt = tf.data.TextLineDataset(tgt_file).map(hi_vectorizer.tokenize)

train_ds = make_dataset(src, tgt, 256)

from modellib import Seq2SeqTransformer
transformer = Seq2SeqTransformer.get(maxlen,
                                     en_vectorizer.vocab_size,
                                     hi_vectorizer.vocab_size)
transformer.summary()
transformer.compile(
    "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

epochs = 10
transformer.fit(train_ds, epochs=epochs)
transformer.save("transformer_epoch10.hdf5")

# transformer = keras.models.load_model("transformer.hdf5")