from __future__ import annotations
from typing import Sequence, Tuple
import tensorflow as tf


def build_cnn_gru(
    input_shape: Tuple[int, int],
    num_classes: int,
    conv_filters: int = 32,
    kernel_size: int = 3,
    pool_size: int = 3,
    gru_units: int = 3,
    dense_units: Sequence[int] = (64, 32),
    dropout_rate: float = 0.2,
) -> tf.keras.Model:

    inputs = tf.keras.layers.Input(shape=input_shape, name="traffic_features")

    x = inputs
    for idx in range(1, 4):
        x = tf.keras.layers.Conv1D(
            filters=conv_filters,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            name=f"conv1d_{idx}",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{idx}")(x)
        x = tf.keras.layers.MaxPooling1D(
            pool_size=pool_size,
            strides=pool_size,
            padding="same",
            name=f"max_pool_{idx}",
        )(x)
    cnn_repr = tf.keras.layers.Flatten(name="cnn_flatten")(x)

    gru_repr = tf.keras.layers.GRU(gru_units, name="gru_branch")(inputs)

    joined = tf.keras.layers.Concatenate(name="concat_cnn_gru")([cnn_repr, gru_repr])
    h = joined
    for idx, units in enumerate(dense_units, start=1):
        h = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{idx}")(h)

    logits = tf.keras.layers.Dense(num_classes, name="class_logits")(h)
    logits = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(logits)
    outputs = tf.keras.layers.Softmax(name="softmax")(logits)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="CNNGRU")
