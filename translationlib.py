import string
from tqdm import tqdm
import json

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