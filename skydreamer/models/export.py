"""Export boundaries for future ONNX/TensorRT support."""

from __future__ import annotations

from pathlib import Path

from skydreamer.models.policy import SkyDreamerPolicy


def export_policy_stub(policy: SkyDreamerPolicy, path: str | Path) -> None:
    """Placeholder export entrypoint.

    The v1 scaffold defines the boundary only. Real export support is deferred
    until model contracts stabilize against simulator data.
    """
    _ = policy
    _ = path
    raise NotImplementedError("Export is intentionally deferred in the v1 scaffold.")

