from tensorflow.keras import layers, models
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = models.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim,
        })
        return config


    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = models.Sequential(
            [layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "latent_dim": self.latent_dim,
        })
        return config

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

class Seq2SeqTransformer:
    @staticmethod
    def get(sequence_length, vocab_size, ):
        embed_dim = 32
        latent_dim = 128
        num_heads = 4

        encoder_inputs = layers.Input(shape=(sequence_length,), dtype="int64", name="encoder_inputs")
        x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
        encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)

        decoder_inputs = layers.Input(shape=(sequence_length,), dtype="int64", name="decoder_inputs")
        encoded_seq_inputs = layers.Input(shape=(sequence_length, embed_dim), name="decoder_state_inputs")
        x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
        x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
        x = layers.Dense(vocab_size, activation='softmax')(x)
        decoder = models.Model([decoder_inputs, encoded_seq_inputs], x)
        decoder.summary()
        decoder_outputs = decoder([decoder_inputs, encoder_outputs])
        transformer = models.Model(
            [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
        )
        return transformer


class Transformer:
    def __init__(self):
        self.model = None
        self.input = None
        self.output = None

    def add_transformer_block(self, hidden_dim, embedding_dim, num_heads):
        attention_head = layers.MultiHeadAttention(num_heads=num_heads,
                                                   key_dim=embedding_dim)(self.output, self.output)

        normalize_1 = layers.LayerNormalization()(attention_head + self.output)
        projection = tf.keras.Sequential(
            [layers.Dense(hidden_dim, activation="relu"), layers.Dense(embedding_dim), ]
        )(normalize_1)
        self.output = layers.LayerNormalization()(normalize_1 + projection)

    def add_text_input(self, input_vocab_size, embedding_dim, seq_len):
        input_layer = layers.Input(shape=(1,), dtype=tf.string)
        self.input = input_layer
        self.vectorizer = TextVectorization(max_tokens=input_vocab_size, output_mode="int",
                                            output_sequence_length=seq_len)
        eng_vectorization = self.vectorizer(input_layer)
        embedding_layer = layers.Embedding(input_dim=input_vocab_size,
                                           output_dim=embedding_dim)(eng_vectorization)

        positions = tf.range(start=0, limit=seq_len, delta=1)
        position_embeddings = layers.Embedding(
            input_dim=seq_len, output_dim=embedding_dim
        )(positions)

        self.output = position_embeddings + embedding_layer

    def add_classifier(self, num_classes):
        x = layers.GlobalAveragePooling1D()(self.output)
        self.output = layers.Dense(num_classes, activation="softmax")(x)

    def build(self):
        self.model = models.Model(inputs=self.input, outputs=self.output)

    def get_vectorizer(self):
        return self.vectorizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    transformer = Transformer()
    transformer.add_text_input(input_vocab_size=30000, embedding_dim=128, seq_len=10)
    transformer.add_transformer_block(hidden_dim=512, embedding_dim=128, num_heads=2)
    transformer.add_transformer_block(hidden_dim=512, embedding_dim=128, num_heads=2)
    transformer.build()
    transformer.model.summary()
