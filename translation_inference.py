import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from modellib import PositionalEmbedding, TransformerDecoder, TransformerEncoder

from wordpiece import SubWordTokenizer

transformer = keras.models.load_model("checkpoints/transformer_epoch10.hdf5",
                                      custom_objects={"PositionalEmbedding": PositionalEmbedding,
                                                      "TransformerEncoder": TransformerEncoder,
                                                      "TransformerDecoder": TransformerDecoder})

maxlen = 25
en_vectorizer = SubWordTokenizer(maxlen=maxlen)
en_vectorizer.load("datasets/parallel/en_new_vocab.txt")

maxlen = 25
hi_vectorizer = SubWordTokenizer(maxlen=maxlen)
hi_vectorizer.load("datasets/parallel/hi_new_vocab.txt")

def decode_sequence(input_sentence):
    tokenized_input_sentence = np.asarray([en_vectorizer.tokenize([input_sentence])])
    decoded_sentence = ""
    tokenized_target_sentence = np.asarray([en_vectorizer.tokenize(decoded_sentence)])
    tokenized_target_sentence[0][1] = en_vectorizer.END

    i = 0
    for i in range(maxlen - 2):
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        tokenized_target_sentence[0][i + 2] = en_vectorizer.END
        tokenized_target_sentence[0][i + 1] = sampled_token_index
        if sampled_token_index == en_vectorizer.END:
            break

    tokenized_target_sentence = tokenized_target_sentence[0][1:i+1]

    decoded_sentence = hi_vectorizer.detokenize(tokenized_target_sentence)[0]
    decoded_sentence = tf.strings.reduce_join(decoded_sentence, separator=' ', axis=-1)
    decoded_sentence = decoded_sentence.numpy().decode()
    return decoded_sentence

tests = ["india has the worst cleaning programme in the world"]

for input_sentence in tests:
    translated = decode_sequence(input_sentence)
    print(input_sentence, translated)

for i in range(1000):
    x = input("English Input >> ")
    ans = decode_sequence(x.strip())
    print(x, ans)
