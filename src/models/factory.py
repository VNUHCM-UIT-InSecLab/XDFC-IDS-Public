from __future__ import annotations
from typing import Any, Dict, Tuple
import tensorflow as tf

from .cnn_gru import build_cnn_gru
from .lstm import build_lstm

def build_model(
    model_name: str,
    input_shape: Tuple[int, int],
    num_classes: int,
    **kwargs: Dict[str, Any],
) -> tf.keras.Model:
    name = model_name.lower().replace("-", "_")
    if name in {"cnn_gru", "cnngru"}:
        return build_cnn_gru(input_shape=input_shape, num_classes=num_classes, **kwargs)
    if name == "lstm":
        return build_lstm(input_shape=input_shape, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unsupported model_name={model_name!r}. Use 'cnn_gru' or 'lstm'.")
