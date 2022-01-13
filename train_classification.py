import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from datasets import load_dataset
from modellib import Transformer
from tensorflow.keras import layers, models, losses, metrics, optimizers
import numpy as np

def create_classifier(num_classes):
    transformer = Transformer()
    transformer.add_text_input(input_vocab_size=10000, embedding_dim=256, seq_len=30)

    transformer.add_transformer_block(hidden_dim=1024, embedding_dim=256, num_heads=3)

    transformer.add_classifier(num_classes=num_classes)
    transformer.build()
    return transformer

dataset = load_dataset("ag_news")
train = dataset["train"]
test = dataset["test"]

train_text = [sample["text"] for sample in train]
print(train_text[:5])
train_labels = [sample["label"] for sample in train]

test_text = [sample["text"] for sample in test]
test_labels = [sample["label"] for sample in test]

classes = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
print(classes)
num_classes = len(classes)

transformer = create_classifier(num_classes)
transformer.model.summary()
transformer.vectorizer.adapt(train_text)
transformer.vectorizer.adapt(test_text)

print(transformer.vectorizer.vocabulary_size())

transformer.model.compile(optimizer=optimizers.Adam(),
                          metrics=[metrics.sparse_categorical_accuracy],
                          loss=losses.sparse_categorical_crossentropy)

transformer.model.fit(train_text, train_labels, batch_size=8, epochs=3, validation_data=(test_text, test_labels))
predictions = transformer.model.predict(test_text, batch_size=1024)

total = 0
incorrect = 0

for index, test in enumerate(test_text):
    total += 1
    label = test_labels[index]
    pred = np.argmax(predictions[index])
    if label != pred:
        incorrect += 1

print(total, incorrect, (total-incorrect)/total)

tests = ["China’s Xi extends support, lauds Kazakhstan’s president for decisive action",
         "TCS to consider share buyback proposal on January 12",
         "Big offers on Samsung foldable smartphones - Galaxy Z Fold 3, Galaxy Z Flip 3 announced - Check details",
         "Novak Djokovic Breaks Silence On Australian Visa Row, Thanks Fans For Support"]

preds = transformer.model.predict(tests)
print(preds)