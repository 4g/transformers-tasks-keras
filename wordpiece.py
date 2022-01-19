import os

import numpy as np
import tensorflow as tf
import pathlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.get_logger().setLevel('ERROR')
pwd = pathlib.Path.cwd()

from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
from tensorflow_text import BertTokenizer, pad_model_inputs


class SpecialTokens:
    PAD = "[PAD]"
    UNK = "[UNK]"
    START = "[START]"
    END = "[END]"
    ALL = [PAD, UNK, START, END]

class Tokenizer:
    def __init__(self, maxlen=25):
        self.vocab = None
        self.special_token_indices = {c: i for i, c in enumerate(SpecialTokens.ALL)}
        self.vocab_size = 0
        self.maxlen = maxlen


    def fit(self, fpath):
        raise NotImplementedError

    def indexof(self, token):
        return self.special_token_indices.get(token, None)

    def save(self, path):
        with open(path, 'w') as fp:
            for token in self.vocab:
                print(token, file=fp)

    def load(self, path):
        raise NotImplementedError

    def add_start(self, tokens):
        return [self.indexof(SpecialTokens.START)] + tokens

    def add_end(self, tokens):
        return tokens + [self.indexof(SpecialTokens.END)]

    def pad(self, tokens, length):
        padding_index = self.indexof(SpecialTokens.PAD)
        pads = np.ones(shape=length, dtype=np.int64) * padding_index
        pads[0:len(tokens)] = tokens
        return pads

class SubWordTokenizer(Tokenizer):
    def __init__(self, maxlen=20):
        super(SubWordTokenizer, self).__init__(maxlen)
        self.bert_tokenizer_params = dict(lower_case=False)
        self.vocab_size = 0
        self.START = tf.constant(self.indexof(SpecialTokens.START), dtype=tf.int64)
        self.END = tf.constant(self.indexof(SpecialTokens.END), dtype=tf.int64)
        self.PAD = tf.constant(self.indexof(SpecialTokens.PAD), dtype=tf.int64)

    def fit(self, text_file_path):
        ds = tf.data.TextLineDataset(text_file_path).map(tf.strings.lower)

        learner_params = dict(lower_thresh=1000,
                              num_iterations=4,
                              max_unique_chars=128,
                              max_token_length=15)

        bert_vocab_args = dict(
            vocab_size=50000,
            reserved_tokens=SpecialTokens.ALL,
            bert_tokenizer_params=self.bert_tokenizer_params,
            learn_params=learner_params,
        )

        self.vocab = bert_vocab.bert_vocab_from_dataset(
            ds.batch(10000).prefetch(2),
            **bert_vocab_args
        )

    def load(self, fpath):
        self.bert_tokenizer = BertTokenizer(fpath, **self.bert_tokenizer_params)
        self.vocab_size = len(list(open(fpath)))

    def tokenize(self, sentence):
        tokens = self.bert_tokenizer.tokenize(sentence)
        tokens = tokens.merge_dims(-2, -1)
        count = tokens.bounding_shape()[0]
        starts = tf.fill([count, 1], self.START)
        ends = tf.fill([count, 1], self.END)
        tokens = tf.concat([starts, tokens, ends], axis=1)


        tokens, mask = pad_model_inputs(input=tokens, max_seq_length=self.maxlen, pad_value=self.indexof(SpecialTokens.PAD))
        tokens = tokens[0]
        return tokens

    def detokenize(self, tokenids):
        return self.bert_tokenizer.detokenize([tokenids])

class CharTokenizer(Tokenizer):
    def __init__(self, maxlen=15):
        super(CharTokenizer, self).__init__(maxlen)
        self.vocab_index = {}
        self.inv_vocab_index = {}
        self.vocab = self.get_vocab()
        self.create_vocab_index()
        self.vocab_size = len(self.vocab_index)
        self.START = self.indexof(SpecialTokens.START)
        self.END = self.indexof(SpecialTokens.END)
        self.PAD = self.indexof(SpecialTokens.PAD)

    def create_vocab_index(self):
        for token in SpecialTokens.ALL:
            self.add_to_vocab(token)

        for token in self.vocab:
            self.add_to_vocab(token)

        self.null_index = self.vocab_index.get(SpecialTokens.UNK)

    def add_to_vocab(self, ch):
        index = len(self.vocab_index)
        self.vocab_index[ch] = index
        self.inv_vocab_index[index] = ch

    def indexof(self, token):
        return self.vocab_index.get(token, self.null_index)

    def tokenize(self, text):
        tokens = self.process(text)
        tokens = [self.indexof(token) for token in tokens]
        tokens = [self.START] + tokens + [self.END]
        padl = [self.PAD for p in range(self.maxlen)]
        tokens = tokens[0:self.maxlen]
        padl[0:len(tokens)] = tokens
        tokens = padl
        if len(tokens) != self.maxlen:
            print(padl)
            # print(len(tokens), self.maxlen, len(padl))
        return tokens

    def detokenize(self, tokens):
        text = ""
        for token in tokens:
            chr = self.inv_vocab_index[token]
            if chr not in SpecialTokens.ALL:
                text += chr
        return text

    @staticmethod
    def get_vocab():
        english_valid_chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
        hi_valid_chars = list([chr(x) for x in range(2304, 2423)])
        all_valid_chars = hi_valid_chars + english_valid_chars
        return all_valid_chars

    def process(self, text):
        text = text.lower()
        return list(filter(lambda x: x in self.vocab, text))


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default=None, help="pass a file to be mapped", required=True)
    parser.add_argument("--vocab", default=None, help="location to save the model", required=True)

    args = parser.parse_args()
    tokenizer = SubWordTokenizer()
    tokenizer.fit(args.text)
    tokenizer.save(args.vocab)