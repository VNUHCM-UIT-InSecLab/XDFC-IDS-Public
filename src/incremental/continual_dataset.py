from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

ArrayPair = Tuple[np.ndarray, np.ndarray]

@dataclass
class ContinualMemory:

    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    seed: int = 42

    def __len__(self) -> int:
        return 0 if self.x is None else int(self.x.shape[0])

    def update(self, x_new: np.ndarray, y_new: np.ndarray, retention_ratio: float) -> ArrayPair:
        retention_ratio = float(np.clip(retention_ratio, 0.0, 1.0))
        rng = np.random.default_rng(self.seed)

        if self.x is None or self.y is None or len(self) == 0:
            self.x = np.asarray(x_new)
            self.y = np.asarray(y_new)
            return self.x, self.y

        keep_size = int(round(retention_ratio * len(self)))
        if keep_size > 0:
            keep_idx = rng.choice(len(self), size=keep_size, replace=False)
            x_old = self.x[keep_idx]
            y_old = self.y[keep_idx]
            self.x = np.concatenate([x_old, np.asarray(x_new)], axis=0)
            self.y = np.concatenate([y_old, np.asarray(y_new)], axis=0)
        else:
            self.x = np.asarray(x_new)
            self.y = np.asarray(y_new)
        return self.x, self.y

    def as_arrays(self) -> ArrayPair:
        if self.x is None or self.y is None:
            raise ValueError("ContinualMemory is empty.")
        return self.x, self.y
