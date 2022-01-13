import random
import os
import sys

import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from enum import Enum
import numpy as np

class SpecialToken(Enum):
    START = "[start]"
    END = "[end]"
    NULL = "[null]"

class CharVectorizer:
    def __init__(self, max_len=15):
        self.max_len = max_len
        self.vocab_index = {}
        self.inv_vocab_index = {}
        self.vocab = self.get_vocab()
        self.special_tokens = {SpecialToken.START, SpecialToken.END, SpecialToken.NULL}
        self.create_vocab_index()
        self.vocab_size = len(self.vocab_index)

    def create_vocab_index(self):
        for token in self.special_tokens:
            self.add_to_vocab(token)

        for token in self.vocab:
            self.add_to_vocab(token)

        self.null_index = self.vocab_index.get(SpecialToken.NULL)

    def add_to_vocab(self, ch):
        index = len(self.vocab_index)
        self.vocab_index[ch] = index
        self.inv_vocab_index[index] = ch

    def indexof(self, token):
        return self.vocab_index.get(token, self.null_index)

    def tokenize(self, text, length):
        nulls = [self.null_index for i in range(length)]
        tokens = self.process(text)
        placeholder = [self.indexof(SpecialToken.START)]
        placeholder += [self.indexof(token) for token in tokens]
        placeholder += [self.indexof(SpecialToken.END)]
        placeholder += nulls
        placeholder = placeholder[:length]
        return placeholder

    def detokenize(self, tokens):
        text = ""
        for token in tokens:
            chr = self.inv_vocab_index[token]
            if chr not in self.special_tokens:
                text += chr
        return text

    @staticmethod
    def get_vocab():
        english_valid_chars = set("abcdefghijklmnopqrstuvwxyz0123456789")
        hi_valid_chars = set([chr(x) for x in range(2304, 2423)])
        all_valid_chars = english_valid_chars.union(hi_valid_chars)
        return all_valid_chars

    def process(self, text):
        text = text.lower()
        return list(filter(lambda x: x in self.vocab, text))

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
    dataset = tf.data.Dataset.from_tensor_slices((src, tgt))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()

def coshuffle(x, y):
    pairs = list(zip(x, y))
    random.shuffle(pairs)
    x, y = zip(*pairs)
    return list(x), list(y)

files = ["datasets/transliterate/dakshina-en-hi.txt", "datasets/transliterate/good_transliteration_pairs.txt"]
src, tgt = [], []
maxlen = 20
vectorizer = CharVectorizer(maxlen)
vocab_size = vectorizer.vocab_size

for f in files:
    x, y = load_data(f)
    src += x
    tgt += y

src = [vectorizer.tokenize(token, maxlen) for token in src]
tgt = [vectorizer.tokenize(token, maxlen + 1) for token in tgt]

src, tgt = coshuffle(src,  tgt)
for i in random.sample(list(zip(src, tgt)), 10):
    s, t = i
    s = vectorizer.detokenize(s)
    t = vectorizer.detokenize(t)
    print(s, t)

split_point = int(len(src) * .85)
train_src, val_src = src[:split_point], src[split_point:]
train_tgt, val_tgt = tgt[:split_point], tgt[split_point:]

train_ds = make_dataset(train_src, train_tgt, 64)
val_ds = make_dataset(val_src, val_tgt, 64)

from modellib import Seq2SeqTransformer
transformer = Seq2SeqTransformer.get(maxlen, vocab_size)
transformer.summary()
transformer.compile(
    "adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

epochs = 2
transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)
transformer.save("transformer.hdf5")

# transformer = keras.models.load_model("transformer.hdf5")

def decode_sequence(input_sentence):
    tokenized_input_sentence = np.asarray([vectorizer.tokenize(input_sentence, length=maxlen)])
    decoded_sentence = ""
    for i in range(maxlen):
        tokenized_target_sentence = np.asarray([vectorizer.tokenize(decoded_sentence, length=maxlen + 1)[:-1]])
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        token = vectorizer.detokenize([sampled_token_index])
        decoded_sentence += token
    return decoded_sentence

tests = ["khana", "samosa", "ladka", "padhai"]

for input_sentence in tests:
    translated = decode_sequence(input_sentence)
    print(input_sentence, translated)

for i in range(1000):
    x = input("English Input >> ")
    tokens = x.strip().split()
    ans = ""
    for token in tokens:
        ans += decode_sequence(token) + " "
    print(x, ans)
