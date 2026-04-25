"""Clock helpers for control-loop timing."""

from __future__ import annotations

import time


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()

