from __future__ import annotations
from typing import List, Sequence
import numpy as np

Weights = List[np.ndarray]

def _normalize_sizes(sizes: Sequence[int]) -> np.ndarray:
    sizes_arr = np.asarray(sizes, dtype=np.float64)
    if sizes_arr.ndim != 1 or len(sizes_arr) == 0:
        raise ValueError("sizes must be a non-empty 1D sequence")
    total = float(np.sum(sizes_arr))
    if total <= 0:
        return np.ones_like(sizes_arr, dtype=np.float64) / len(sizes_arr)
    return sizes_arr / total

def aggregate_deltas(base_weights: Weights, deltas: Sequence[Weights], sizes: Sequence[int]) -> Weights:
    coeffs = _normalize_sizes(sizes)
    aggregated: Weights = []
    for base_layer, delta_layers in zip(base_weights, zip(*deltas)):
        layer_delta = np.zeros_like(base_layer, dtype=np.float32)
        for coeff, delta in zip(coeffs, delta_layers):
            layer_delta += coeff * delta.astype(np.float32)
        aggregated.append(base_layer.astype(np.float32) - layer_delta)
    return aggregated
