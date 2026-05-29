from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

@dataclass
class ClassExplanation:
    class_id: int
    class_name: str
    top_features: List[str]
    top_indices: List[int]
    trustworthiness: float
    drift_l2: Optional[float]
    mean_attribution: np.ndarray

@dataclass
class LocalSHAPAnalyzer:
    background_size: int = 100
    explain_size: int = 200
    top_k: int = 10
    previous_class_vectors: Dict[int, np.ndarray] = field(default_factory=dict)

    def _sample_background(self, x: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        n_bg = min(self.background_size, len(x))
        n_explain = min(self.explain_size, len(x))
        bg_idx = rng.choice(len(x), size=n_bg, replace=False)
        ex_idx = rng.choice(len(x), size=n_explain, replace=False)
        return x[bg_idx].astype(np.float32), x[ex_idx].astype(np.float32)

    def _deep_shap(self, model: tf.keras.Model, x_bg: np.ndarray, x_exp: np.ndarray) -> Optional[np.ndarray]:
        try:
            import shap
            try:
                explainer = shap.DeepExplainer(model, x_bg)
                values = explainer.shap_values(x_exp)
            except Exception:
                explainer = shap.GradientExplainer(model, x_bg)
                values = explainer.shap_values(x_exp)

            if isinstance(values, list):
                stacked = np.stack([np.squeeze(v) for v in values], axis=1)
            else:
                arr = np.asarray(values)
                arr = np.squeeze(arr)
                if arr.ndim == 3 and arr.shape[-1] == model.output_shape[-1]:
                    stacked = np.transpose(arr, (0, 2, 1))
                elif arr.ndim == 3:
                    stacked = arr
                else:
                    return None
            return stacked.astype(np.float32)
        except Exception:
            return None

    def _gradient_times_input(self, model: tf.keras.Model, x_exp: np.ndarray) -> np.ndarray:
        x_tensor = tf.convert_to_tensor(x_exp.astype(np.float32))
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            probs = model(x_tensor, training=False)
        num_classes = int(probs.shape[-1])
        attrs = []
        for cls in range(num_classes):
            with tf.GradientTape() as class_tape:
                class_tape.watch(x_tensor)
                cls_probs = model(x_tensor, training=False)[:, cls]
            grads = class_tape.gradient(cls_probs, x_tensor)
            attrs.append(np.squeeze((grads * x_tensor).numpy()))
        return np.stack(attrs, axis=1).astype(np.float32)

    def explain(
        self,
        model: tf.keras.Model,
        x_val: np.ndarray,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        seed: int = 42,
    ) -> Dict[int, ClassExplanation]:
        if len(x_val) == 0:
            raise ValueError("x_val is empty; cannot compute local explanations.")
        feature_names = feature_names or [f"feature_{i}" for i in range(x_val.shape[1])]
        num_classes = int(model.output_shape[-1])
        class_names = class_names or [f"class_{i}" for i in range(num_classes)]

        x_bg, x_exp = self._sample_background(x_val, seed=seed)
        probs = model.predict(x_exp.astype(np.float32), verbose=0)
        y_pred = np.argmax(probs, axis=1)

        all_attr = self._deep_shap(model, x_bg, x_exp)
        if all_attr is None:
            all_attr = self._gradient_times_input(model, x_exp)

        explanations: Dict[int, ClassExplanation] = {}
        for class_id in sorted(np.unique(y_pred).astype(int).tolist()):
            idx = np.where(y_pred == class_id)[0]
            if len(idx) == 0:
                continue
            class_attr = all_attr[idx, class_id, :]
            mean_attr = np.mean(class_attr, axis=0)
            ranking = np.argsort(np.abs(mean_attr))[::-1]
            top_idx = ranking[: min(self.top_k, len(ranking))].astype(int).tolist()
            top_features = [feature_names[i] if i < len(feature_names) else f"feature_{i}" for i in top_idx]
            trust = float(np.mean(np.abs(mean_attr[top_idx]))) if top_idx else 0.0
            prev = self.previous_class_vectors.get(class_id)
            drift = float(np.linalg.norm(mean_attr - prev, ord=2)) if prev is not None else None
            explanations[class_id] = ClassExplanation(
                class_id=class_id,
                class_name=class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                top_features=top_features,
                top_indices=top_idx,
                trustworthiness=trust,
                drift_l2=drift,
                mean_attribution=mean_attr,
            )
            self.previous_class_vectors[class_id] = mean_attr
        return explanations

    @staticmethod
    def to_frame(explanations: Dict[int, ClassExplanation]) -> pd.DataFrame:
        rows = []
        for item in explanations.values():
            rows.append(
                {
                    "class_id": item.class_id,
                    "class_name": item.class_name,
                    "top_features": ", ".join(item.top_features),
                    "top_indices": ", ".join(map(str, item.top_indices)),
                    "trustworthiness": item.trustworthiness,
                    "drift_l2": item.drift_l2,
                }
            )
        return pd.DataFrame(rows)
