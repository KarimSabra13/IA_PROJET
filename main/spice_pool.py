from __future__ import annotations

import itertools
from typing import Any, Dict, List, Tuple

from .spice_worker import PyngsWorker


class PyngsWorkerPool:
    def __init__(self, n_workers: int = 4, **worker_kwargs) -> None:
        self.workers = [PyngsWorker(**worker_kwargs) for _ in range(n_workers)]
        self._rr = itertools.cycle(range(n_workers))

    def measure(self, wn: float, wp: float, **kwargs) -> Dict[str, Any]:
        i = next(self._rr)
        return self.workers[i].measure(wn, wp, **kwargs)

    def close(self) -> None:
        for w in self.workers:
            w.close()
