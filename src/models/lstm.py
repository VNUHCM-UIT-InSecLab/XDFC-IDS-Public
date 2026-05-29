from __future__ import annotations
from typing import Optional, Tuple
import tensorflow as tf

def build_lstm(
    input_shape: Tuple[int, int],
    num_classes: int,
    lstm_units: int = 50,
    hidden_units: Optional[int] = None,
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape, name="traffic_features")
    h = tf.keras.layers.LSTM(lstm_units, name="lstm_encoder")(inputs)
    if hidden_units is not None and hidden_units > 0:
        h = tf.keras.layers.Dense(hidden_units, activation="relu", name="dense_hidden")(h)
    logits = tf.keras.layers.Dense(num_classes, name="class_logits")(h)
    logits = tf.keras.layers.Dropout(dropout_rate, name="classifier_dropout")(logits)
    outputs = tf.keras.layers.Softmax(name="softmax")(logits)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="LSTM")
