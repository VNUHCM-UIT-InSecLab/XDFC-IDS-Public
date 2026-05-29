from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from incremental.continual_dataset import ContinualMemory
from incremental.ewc import EWC
from utils.metrics import evaluate_model

Weights = List[np.ndarray]

@dataclass
class Collaborator:
    collaborator_id: int
    model: tf.keras.Model
    learning_rate: float = 0.01
    ewc_lambda: float = 1e-3
    seed: int = 42
    validation_split: float = 0.2
    optimizer: tf.keras.optimizers.Optimizer = field(init=False)
    ewc: EWC = field(init=False)
    memory: ContinualMemory = field(init=False)
    x_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    x_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.ewc = EWC(lambda_=self.ewc_lambda)
        self.memory = ContinualMemory(seed=self.seed + self.collaborator_id)

    @property
    def train_size(self) -> int:
        return 0 if self.x_train is None else int(self.x_train.shape[0])

    def get_weights(self) -> Weights:
        return [w.copy() for w in self.model.get_weights()]

    def set_weights(self, weights: Weights) -> None:
        self.model.set_weights(weights)

    def acquire_task_data(
        self,
        x_new: np.ndarray,
        y_new: np.ndarray,
        retention_ratio: float,
    ) -> None:
        if x_new.size == 0:
            if len(self.memory) == 0:
                self.x_train = self.y_train = self.x_val = self.y_val = None
                return
            x_all, y_all = self.memory.as_arrays()
        else:
            x_all, y_all = self.memory.update(x_new, y_new, retention_ratio)

        if len(y_all) < 2 or len(np.unique(y_all)) < 2:
            self.x_train, self.y_train = x_all, y_all
            self.x_val, self.y_val = x_all, y_all
            return

        stratify = y_all if np.min(np.bincount(y_all.astype(int))) >= 2 else None
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            x_all,
            y_all,
            test_size=self.validation_split,
            random_state=self.seed + self.collaborator_id,
            stratify=stratify,
        )

    def make_dataset(self, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> tf.data.Dataset:
        ds = tf.data.Dataset.from_tensor_slices((x.astype(np.float32), y.astype(np.int64)))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(y), 10000), seed=self.seed + self.collaborator_id)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def train_local(
        self,
        epochs: int,
        batch_size: int,
        use_ewc: bool,
    ) -> Dict[str, float]:
        if self.x_train is None or self.y_train is None or self.train_size == 0:
            return {"loss": 0.0, "ce_loss": 0.0, "ewc_loss": 0.0}

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        ds = self.make_dataset(self.x_train, self.y_train, batch_size=batch_size, shuffle=True)
        last_logs = {"loss": 0.0, "ce_loss": 0.0, "ewc_loss": 0.0}

        for _ in range(epochs):
            for x_batch, y_batch in ds:
                with tf.GradientTape() as tape:
                    preds = self.model(x_batch, training=True)
                    ce_loss = loss_fn(y_batch, preds)
                    ewc_loss = self.ewc.penalty(self.model) if use_ewc and self.ewc.is_ready else 0.0
                    total_loss = ce_loss + ewc_loss
                grads = tape.gradient(total_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                last_logs = {
                    "loss": float(total_loss.numpy()),
                    "ce_loss": float(ce_loss.numpy()),
                    "ewc_loss": float(ewc_loss.numpy() if hasattr(ewc_loss, "numpy") else ewc_loss),
                }
        return last_logs

    def compute_delta(self, base_weights: Weights) -> Weights:
        return [base - current for current, base in zip(self.get_weights(), base_weights)]

    def consolidate(self, batch_size: int, fisher_batches: Optional[int] = None) -> None:
        if self.x_train is None or self.y_train is None or self.train_size == 0:
            return
        ds = self.make_dataset(self.x_train, self.y_train, batch_size=batch_size, shuffle=False)
        self.ewc.estimate_fisher(self.model, ds, max_batches=fisher_batches)

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        return evaluate_model(self.model, x, y)
