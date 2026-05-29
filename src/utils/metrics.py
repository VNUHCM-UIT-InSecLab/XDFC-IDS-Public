from __future__ import annotations
from typing import Dict
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate_model(
    model: tf.keras.Model,
    x: np.ndarray,
    y: np.ndarray,
    average: str = "weighted",
    batch_size: int = 4096,
) -> Dict[str, float]:
    probs = model.predict(x.astype(np.float32), batch_size=batch_size, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y, y_pred, average=average, zero_division=0)),
    }

def delta_from_task1(task1_acc: float, current_acc: float) -> float:
    """Delta_1"""
    return float(task1_acc - current_acc)

def delta_against_xdfc(xdfc_acc: float, other_acc: float) -> float:
    """Delta_2"""
    return float(xdfc_acc - other_acc)
