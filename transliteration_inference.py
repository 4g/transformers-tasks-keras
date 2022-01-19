import os
import modellib
from wordpiece import CharTokenizer
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
tf.get_logger().setLevel('ERROR')

import numpy as np

transformer = modellib.load("checkpoints/transliterator_transformer.hdf5")

maxlen = 20
src_vectorizer = CharTokenizer(maxlen)
tgt_vectorizer = CharTokenizer(maxlen)

vocab_size = src_vectorizer.vocab_size

def decode_sequence(input_sentence):
    tokenized_input_sentence = np.asarray([src_vectorizer.tokenize(input_sentence)])
    decoded_sentence = ""
    tokenized_target_sentence = np.asarray([tgt_vectorizer.tokenize(decoded_sentence)])
    tokenized_target_sentence[0][1] = src_vectorizer.END

    i = 0
    for i in range(maxlen - 2):
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        tokenized_target_sentence[0][i + 2] = src_vectorizer.END
        tokenized_target_sentence[0][i + 1] = sampled_token_index
        if sampled_token_index == src_vectorizer.END:
            break

    tokenized_target_sentence = tokenized_target_sentence[0][1:i+1]
    decoded_sentence = tgt_vectorizer.detokenize(tokenized_target_sentence)
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
    print(ans)
