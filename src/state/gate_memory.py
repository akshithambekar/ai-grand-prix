"""Course/gate memory skeleton.

VQ1 gates are consistent within the qualifier, but online visual confirmation
should remain the source of truth for reliability.
"""

from __future__ import annotations

from perception.observations import GateObservation


class GateMemory:
    def remember(self, gate: GateObservation) -> None:
        raise NotImplementedError

