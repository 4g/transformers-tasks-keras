import string
from tqdm import tqdm
import json
from enum import Enum

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


class LangTransform:
    def __init__(self, num_chars=52):
        self.charmap = {}
        self.reverse_charmap = {}
        self.num_chars = num_chars
        self.usable_chars = string.printable[:num_chars]
        self.special_char = " "

    def save(self, filepath):
        with open(filepath, 'w') as fp:
            print("Saving model to ", filepath)
            json.dump(self.__dict__, fp)

    def load(self, filepath):
        with open(filepath, 'r') as fp:
            print("Loading the model from ", filepath)
            self.__dict__ = json.load(fp)

    def adapt(self, texts):
        char_hist = {}
        for text in tqdm(texts, desc="Learning char transform ... "):
            for word in text.split():
                for c in word:
                    char_hist[c] = char_hist.get(c, 0) + 1

        sorted_chars = sorted(char_hist, key=char_hist.get, reverse=True)
        for index, c in enumerate(sorted_chars[:self.num_chars]):
            c_ = self.usable_chars[index]
            self.charmap[c] = c_
            self.reverse_charmap[c_] = c

    def transform(self, text):
        final_string = ""
        for word in text.split():
            for c in word:
                final_string += self.charmap.get(c, self.special_char)
            final_string += " "
        final_string = final_string.strip()
        return final_string

    def reverse_transform(self, text):
        final_string = ""
        for word in text.split():
            for c in word:
                final_string += self.reverse_charmap.get(c, self.special_char)
            final_string += " "
        final_string = final_string.strip()
        return final_string


if __name__ == "__main__":
    import argparse, os, glob, traceback
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="pass a file to be mapped", required=True)
    parser.add_argument("--model", default=None, help="location to save the model", required=True)

    args = parser.parse_args()

    hi = [i.strip() for i in open(args.input)]

    lt = LangTransform()
    lt.adapt(hi)
    lt.save(args.model)

    lt_load = LangTransform()
    lt_load.load(args.model)

    from random import sample
    for i in sample(hi, 100):
        j = lt_load.transform(i)
        it = lt_load.reverse_transform(j)
        print(i, j, it)