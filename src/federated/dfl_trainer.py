from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from federated.aggregation import aggregate_deltas
from federated.collaborator import Collaborator

Weights = List[np.ndarray]
TaskData = Sequence[Tuple[np.ndarray, np.ndarray]]
ModelBuilder = Callable[[], tf.keras.Model]


@dataclass
class XDFCIDSConfig:
    num_collaborators: int
    communication_rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 1024
    learning_rate: float = 0.01
    retention_ratio: float = 0.6
    ewc_lambda: float = 1e-3
    validation_split: float = 0.2
    fisher_batches: Optional[int] = 10
    seed: int = 42


@dataclass
class DecentralizedFederatedTrainer:
    model_builder: ModelBuilder
    config: XDFCIDSConfig
    collaborators: List[Collaborator] = field(init=False)

    def __post_init__(self) -> None:
        tf.keras.utils.set_random_seed(self.config.seed)
        base_model = self.model_builder()
        base_weights = base_model.get_weights()
        self.collaborators = []
        for idx in range(self.config.num_collaborators):
            model = self.model_builder()
            model.set_weights(base_weights)
            self.collaborators.append(
                Collaborator(
                    collaborator_id=idx,
                    model=model,
                    learning_rate=self.config.learning_rate,
                    ewc_lambda=self.config.ewc_lambda,
                    seed=self.config.seed,
                    validation_split=self.config.validation_split,
                )
            )

    def expand_collaborators(self, new_total: int) -> None:
        if new_total <= len(self.collaborators):
            return
        reference_weights = self.consensus_weights()
        for idx in range(len(self.collaborators), new_total):
            model = self.model_builder()
            model.set_weights(reference_weights)
            self.collaborators.append(
                Collaborator(
                    collaborator_id=idx,
                    model=model,
                    learning_rate=self.config.learning_rate,
                    ewc_lambda=self.config.ewc_lambda,
                    seed=self.config.seed,
                    validation_split=self.config.validation_split,
                )
            )
        self.config.num_collaborators = new_total

    def consensus_weights(self) -> Weights:
        return self.collaborators[0].get_weights()

    def train_task(
        self,
        task_id: int,
        task_data: TaskData,
        retention_ratio: Optional[float] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        
        epsilon = self.config.retention_ratio if retention_ratio is None else retention_ratio
        if len(task_data) > len(self.collaborators):
            self.expand_collaborators(len(task_data))

        for col, (x_new, y_new) in zip(self.collaborators, task_data):
            col.acquire_task_data(x_new, y_new, retention_ratio=epsilon)

        round_iter = range(self.config.communication_rounds)
        if verbose:
            round_iter = tqdm(round_iter, desc=f"Task {task_id} DFL rounds", leave=False)

        logs: Dict[str, float] = {}
        for _round in round_iter:
            base_weights = self.consensus_weights()
            sizes: List[int] = []
            deltas: List[Weights] = []
            losses: List[float] = []

            for col in self.collaborators:
                col.set_weights(base_weights)
                local_logs = col.train_local(
                    epochs=self.config.local_epochs,
                    batch_size=self.config.batch_size,
                    use_ewc=(task_id > 1),
                )
                losses.append(local_logs.get("loss", 0.0))
                sizes.append(max(col.train_size, 0))
                deltas.append(col.compute_delta(base_weights))

            new_weights = aggregate_deltas(base_weights=base_weights, deltas=deltas, sizes=sizes)
            for col in self.collaborators:
                col.set_weights(new_weights)

            logs = {
                "mean_train_loss": float(np.mean(losses)) if losses else 0.0,
                "total_train_samples": float(np.sum(sizes)),
            }

        for col in self.collaborators:
            col.consolidate(batch_size=self.config.batch_size, fisher_batches=self.config.fisher_batches)
        return logs

    def evaluate_all(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        metrics = [col.evaluate(x, y) for col in self.collaborators]
        out: Dict[str, float] = {}
        for key in metrics[0].keys():
            out[key] = float(np.mean([m[key] for m in metrics]))
        return out
