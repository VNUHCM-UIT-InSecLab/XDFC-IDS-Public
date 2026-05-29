from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple
import tensorflow as tf

Batch = Tuple[tf.Tensor, tf.Tensor]

@dataclass
class EWC:
    lambda_: float = 1e-3
    fisher_diagonal: Optional[List[tf.Tensor]] = field(default=None, init=False)
    reference_weights: Optional[List[tf.Tensor]] = field(default=None, init=False)

    @property
    def is_ready(self) -> bool:
        return self.fisher_diagonal is not None and self.reference_weights is not None

    def snapshot(self, model: tf.keras.Model) -> None:
        self.reference_weights = [tf.identity(v) for v in model.trainable_variables]

    def estimate_fisher(
        self,
        model: tf.keras.Model,
        dataset: Iterable[Batch],
        max_batches: Optional[int] = None,
    ) -> None:

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        fisher = [tf.zeros_like(v, dtype=tf.float32) for v in model.trainable_variables]
        n_batches = 0

        for batch_id, (x_batch, y_batch) in enumerate(dataset):
            if max_batches is not None and batch_id >= max_batches:
                break
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=False)
                loss = loss_fn(y_batch, preds)
            grads = tape.gradient(loss, model.trainable_variables)
            for i, grad in enumerate(grads):
                if grad is not None:
                    fisher[i] = fisher[i] + tf.square(tf.cast(grad, tf.float32))
            n_batches += 1

        denom = tf.cast(max(n_batches, 1), tf.float32)
        self.fisher_diagonal = [f / denom for f in fisher]
        self.snapshot(model)

    def penalty(self, model: tf.keras.Model) -> tf.Tensor:
        if not self.is_ready:
            return tf.constant(0.0, dtype=tf.float32)
        total = tf.constant(0.0, dtype=tf.float32)
        assert self.fisher_diagonal is not None
        assert self.reference_weights is not None
        for var, fisher, ref in zip(model.trainable_variables, self.fisher_diagonal, self.reference_weights):
            total = total + tf.reduce_sum(fisher * tf.square(tf.cast(var, tf.float32) - ref))
        return tf.cast(self.lambda_, tf.float32) * 0.5 * total
